"""音声ファイルを読み込み文字起こしを行う
"""

import csv
import gc
import math
import platform
import subprocess
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import filedialog

import librosa
import numpy as np
import PySimpleGUI as sg
import pytz
import torch
import whisper
from pydantic.dataclasses import Field, dataclass
from tqdm import tqdm

torch.cuda.is_available = lambda: False

#######################################################################################


def main():
    CONFIGS = Configs()
    DIRECTORY_PATH = DirectoryPath()
    DIRECTORY_PATH.make_all_directories()

    (model_name, file_path) = get_model_name_file_path_from_gui(
        model_description=CONFIGS.model_description,
        initial_model_name=CONFIGS.initial_model_name,
    )

    print()
    print(f"モデル名： {model_name}")
    print(f"音声ファイルのパス： {file_path}")

    # openai-whisper は 30 秒ごとのデータに対して演算を行うため，30 秒間のデータ長さを取得する
    sampling_rate = librosa.get_samplerate(path=file_path)
    print(f"    サンプリングレート： {sampling_rate} [Hz]")

    model = whisper.load_model(name=model_name, download_root=DIRECTORY_PATH.model)

    audio = whisper.load_audio(file=file_path)
    audio_time = len(audio) / sampling_rate

    start_time = datetime.now(pytz.timezone(zone="Asia/Tokyo"))
    end_time = start_time + timedelta(
        seconds=int(audio_time * CONFIGS.name_sec[model_name])
    )
    print(f"    再生時間： {audio_time:.3f} [s]")
    print(f"       変換開始時刻時刻       ： {start_time}")
    print(f"    -> 変換完了予定時刻 (参考)： {end_time}")

    print("    音声からテキストを推論中．．．")
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=UserWarning)
        voice_to_txt(
            model=model,
            audio=audio,
            sampling_rate=sampling_rate,
            file_path=file_path,
            model_name=model_name,
            log_dir=DIRECTORY_PATH.log,
        )

    result_writer = ResultWriter(
        file_path=file_path,
        model_name=model_name,
        log_dir=DIRECTORY_PATH.log,
        output_dir=DIRECTORY_PATH.output,
    )
    transcribed_text: str = result_writer.get_transcribed_text()
    result_writer.save_transcribed_text(text=transcribed_text)

    open_on_explorer(dir_path=DIRECTORY_PATH.output)


#######################################################################################


@dataclass(frozen=True)
class Configs(object):
    model_description: dict[str, str] = Field(
        default_factory=lambda: {
            "tiny": "低精度／超高速 (large の 32 倍速, 60 s の変換に 約 20 s)",
            "base": "低精度／準高速 (large の 16 倍速, 60 s の変換に 約 40 s)",
            "small": "並精度／並速 (large の 6 倍速, 60 s の変換に 約 70 s)",
            "medium": "高精度／低速 (large の 2 倍速, 60 s の変換に 約 300 s)",
            "large": "超高精度／超低速 (16 GB メモリ／CPU のみだと実行不可)",
        }
    )
    name_sec: dict[str, str] = Field(
        default_factory=lambda: {
            "tiny": 20 / 60,
            "base": 40 / 60,
            "small": 70 / 60,
            "medium": 300 / 60,
            "large": 1000 / 60,
        }
    )
    initial_model_name: str = "small"


@dataclass(frozen=True)
class DirectoryPath(object):
    model: Path = Path("../.model")
    input: Path = Path("../data/input")
    output: Path = Path("../data/output")
    log: Path = Path("../data/log")

    def make_all_directories(self):
        """ディレクトリ作成．"""
        self.model.mkdir(parents=True, exist_ok=True)
        self.input.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)
        self.log.mkdir(parents=True, exist_ok=True)


class CsvLogger(object):
    def __init__(self, file_path: str, model_name: str, log_dir: Path) -> None:
        """変換時に中断しても次回以降に途中から変換を開始できるように，
        30 [s] ごとに変換された文字列のログを取るためのロガー．

        Args:
            file_path (str): 音声ファイルのパス．
            model_name (str): モデル名称．
            log_dir (Path): ログ保存先．
        """
        self._log_path: Path = log_dir / f"{Path(file_path).stem}_{model_name}.csv"
        print(f"ログファイル： {self._log_path}")

    @property
    def log_path(self):
        path = self._log_path
        assert path.exists(), FileNotFoundError()
        return path

    def get_restart_index(self) -> int:
        if not self._log_path.exists():
            return 0

        with self._log_path.open(mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            lines = [r for r in reader]
        restart_index = len(lines) - 1
        return restart_index

    def write_log(self, transcribed: dict):
        if not self._log_path.exists():
            with self._log_path.open(mode="w", encoding="utf-8", newline="\n") as f:
                writer = csv.writer(f)
                writer.writerow(["language", "text", "segments"])
                writer.writerow(
                    [
                        transcribed["language"],
                        transcribed["text"],
                        transcribed["segments"],
                    ]
                )
        else:
            with self._log_path.open(mode="a", encoding="utf-8", newline="\n") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        transcribed["language"],
                        transcribed["text"],
                        transcribed["segments"],
                    ]
                )


class ResultWriter(object):
    def __init__(
        self, file_path: str, model_name: str, log_dir: Path, output_dir: Path
    ) -> None:
        """ログ保存された .csv ファイルから文字起こしされた出力の .txt ファイルを生成する．

        Args:
            file_path (str): 音声ファイルパス．
            model_name (str): モデル名称．
            log_dir (Path): ログ保存先．
            output_dir (Path): 文字起こしされた .txt ファイルの出力先．
        """
        self._log_path: Path = log_dir / f"{Path(file_path).stem}_{model_name}.csv"
        self._save_path: Path = output_dir / f"{Path(file_path).stem}_{model_name}.txt"
        print(f"保存先： {self._save_path}")

    def get_transcribed_text(self, key_name: str = "text") -> str:
        with self._log_path.open(mode="r", encoding="utf-8") as f:
            _reader = csv.DictReader(f)
            _text_lines = [r[key_name] for r in _reader]
        text = "".join(_text_lines)
        return text

    def save_transcribed_text(self, text: str):
        with self._save_path.open(mode="w", encoding="utf-8") as f:
            f.write(text)


#######################################################################################


def get_file_path_on_explorer() -> str:
    """エクスプローラーから音声ファイルを選択する．

    Returns:
        str: 音声ファイルのパス
    """
    filetypes = [("音声ファイル", "*.mp3 *.m4a *.wave *.aif *.aac *.flac")]
    initial_dir = "../data/input"
    file_path = filedialog.askopenfilename(
        filetypes=filetypes, initialdir=initial_dir, title="音声ファイルを選択してください"
    )
    if file_path:
        return file_path
    else:
        return


def about_models_text(model_description: dict[str, str]) -> list[sg.Text]:
    about_models_list = [
        [
            sg.Text(
                "CPU: Intel Core i5-1135G7 / メモリ: 16 GB の場合の参考速度",
                font=("BIZ UD Gothic", 14),
            )
        ]
    ]
    for k, v in model_description.items():
        about_models_list.append(
            [sg.Text(f"    {k}: {v}", text_color="cyan", font=("BIZ UD Gothic", 14))]
        )
    return about_models_list


def get_model_name_file_path_from_gui(
    model_description: dict[str, str], initial_model_name: str
) -> tuple[str, str]:
    """GUI の結果からモデル名称とファイルパスを取得する．

    Args:
        model_description (dict[str, str]): モデルの説明．
        initial_model_name (str): 初期に選択されるモデル名称．

    Returns:
        tuple[str, str]: モデル名称とファイルパスのタプル．
    """
    sg.theme("DarkGrey3")

    file_path = None

    layout = [
        [
            sg.Text(text="右のプルダウンからモデル名称を選択してください： ", font=("BIZ UD Gothic", 14)),
            sg.Combo(
                values=list(model_description.keys()),
                default_value=initial_model_name,
                readonly=True,
                font=("BIZ UD Gothic", 14),
            ),
        ],
    ]
    layout += about_models_text(model_description=model_description)
    layout += [[sg.Text(text="", font=("BIZ UD Gothic", 14))]]
    layout += [
        [
            sg.Text(text="右のボタンを押して音声ファイルを選んでください： ", font=("BIZ UD Gothic", 14)),
            sg.Button(button_text="音声ファイルを選択", font=("BIZ UD Gothic", 14)),
        ],
        [
            sg.Text(
                text=f"    ファイルパス： {file_path}",
                key="file_path",
                text_color="orange",
                font=("BIZ UD Gothic", 14),
            )
        ],
        [
            sg.Button("OK", font=("BIZ UD Gothic", 14)),
            sg.Button("キャンセル", font=("BIZ UD Gothic", 14)),
        ],
    ]

    window = sg.Window(title="voice2txt", layout=layout, font=("BIZ UD Gothic", 14))

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or (event == "キャンセル"):
            print("\nキャンセルされたので処理を終了しました．\n")
            window.close()
            exit()

        elif event == "音声ファイルを選択":
            file_path: str = get_file_path_on_explorer()
            window["file_path"].Update(file_path)

        elif event == "OK":
            if not file_path:
                continue
            elif not values:
                continue
            else:
                break

        else:
            raise ValueError()

    window.close()

    model_name = list(values.values())[0]

    return (model_name, file_path)


def voice_to_txt(
    model: whisper.model.Whisper,
    audio: np.ndarray,
    sampling_rate: float,
    file_path: str,
    model_name: str,
    log_dir: Path,
):
    """音声データから文字起こしを行う．

    Args:
        model (whisper.model.Whisper): モデル．
        audio (np.ndarray): 音声データ．
        sampling_rate (float): サンプリングレート．
        file_path (str): 入力となる音声ファイルのパス．
        model_name (str): モデル名称．
        log_dir (Path): ログ保存先．
    """
    csv_logger = CsvLogger(
        file_path=file_path,
        model_name=model_name,
        log_dir=log_dir,
    )
    length_30_sec = 30 * sampling_rate
    restart_index = csv_logger.get_restart_index()
    sequences = math.ceil(len(audio) / length_30_sec)

    if sequences - restart_index < 1e-7:
        print("このモデルとファイルの組み合わせは，すでに文字起こし処理済みです．")
        return

    for idx_30_sec in tqdm(range(restart_index, sequences), dynamic_ncols=True):
        start_idx = idx_30_sec * length_30_sec
        end_idx = (idx_30_sec + 1) * length_30_sec

        if end_idx > len(audio):
            audio_30_sec = audio[start_idx:]
        else:
            audio_30_sec = audio[start_idx:end_idx]

        try:
            transcribed = model.transcribe(audio=audio_30_sec, fp16=True)
            csv_logger.write_log(transcribed=transcribed)

        except Exception:
            continue

        gc.collect()


def open_on_explorer(dir_path: Path):
    """出力保存先をエクスプローラーで開く．

    Args:
        dir_path (Path): エクスプローラーで開きたいフォルダのパス．
    """
    if platform.system() != "Windows":
        return
    subprocess.Popen(["explorer", str(dir_path)], shell=True)


#######################################################################################

if __name__ == "__main__":
    main()
    exit()
