#!/usr/bin/env python3
"""
Boucle vocale GLaDOS locale :
 - Capture micro
 - Transcription Whisper (faster-whisper)
 - LLM via Ollama
 - Synthèse vocale GLaDOS via Piper
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

GLaDOS_SYSTEM_PROMPT = (
    "You are GLaDOS from Portal. You answer concisely, with a dry, sarcastic, "
    "matter-of-fact tone. Never apologize. Stay in character."
)


class PiperGladosTTS:
    """
    Utilise le binaire `piper` + un modèle GLaDOS (.onnx + .json) pour synthétiser en WAV.
    """

    def __init__(
        self,
        model_path: Path,
        config_path: Optional[Path] = None,
        speaker: Optional[int] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
    ) -> None:
        self.binary = shutil.which("piper")
        if not self.binary:
            raise RuntimeError("binaire `piper` introuvable. Installe-le et assure-toi qu'il est dans le PATH.")

        self.model_path = Path(model_path).expanduser()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle Piper introuvable : {self.model_path}")

        if config_path:
            self.config_path = Path(config_path).expanduser()
        else:
            # Convention des voix piper : <model>.onnx.json
            self.config_path = Path(str(self.model_path) + ".json")

        if not self.config_path.exists():
            raise FileNotFoundError(f"Fichier de config Piper introuvable : {self.config_path}")

        self.speaker = speaker
        self.length_scale = length_scale
        self.noise_scale = noise_scale

    def synthesize(self, text: str, destination: Optional[Path] = None) -> Path:
        """
        Génère un WAV et renvoie son chemin.
        """
        if not text.strip():
            raise ValueError("Texte vide pour la synthèse.")

        if destination is None:
            fd, wav_path = tempfile.mkstemp(prefix="glados_", suffix=".wav")
            os.close(fd)
            destination = Path(wav_path)
        else:
            destination = Path(destination)

        cmd = [
            self.binary,
            "--model",
            str(self.model_path),
            "--config",
            str(self.config_path),
            "--output_file",
            str(destination),
        ]

        if self.speaker is not None:
            cmd += ["--speaker", str(self.speaker)]
        if self.length_scale is not None:
            cmd += ["--length_scale", str(self.length_scale)]
        if self.noise_scale is not None:
            cmd += ["--noise_scale", str(self.noise_scale)]

        try:
            subprocess.run(cmd, input=text.encode("utf-8"), check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Echec Piper : {exc}") from exc

        return destination


class OllamaClient:
    """
    Client simple pour Ollama (localhost par défaut).
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        system_prompt: str = GLaDOS_SYSTEM_PROMPT,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.system_prompt = system_prompt
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "system": self.system_prompt,
            "prompt": prompt,
            "stream": False,
        }
        url = f"{self.host}/api/generate"
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Ollama a renvoyé une erreur : {data['error']}")
        return data.get("response", "").strip()


class WhisperSTT:
    """
    Wrapper minimal autour de faster-whisper.
    """

    def __init__(self, model_name: str, device: str = "cpu", compute_type: str = "int8", language: Optional[str] = None):
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.language = language

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, language=self.language, beam_size=1)
        text = "".join(seg.text for seg in segments).strip()
        return text


def record_audio(seconds: float, samplerate: int) -> np.ndarray:
    sd.default.samplerate = samplerate
    sd.default.channels = 1
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def play_wav(path: Path) -> None:
    data, sr = sf.read(path, dtype="float32")
    sd.play(data, sr)
    sd.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Boucle vocale GLaDOS (micro -> Whisper -> Ollama -> Piper).")
    parser.add_argument("--listen-seconds", type=float, default=3.0, help="Durée d'enregistrement micro par tour.")
    parser.add_argument("--samplerate", type=int, default=16000, help="Fréquence d'échantillonnage micro.")
    parser.add_argument("--language", type=str, default=None, help="Code langue Whisper (ex: en, fr). None = auto.")
    parser.add_argument("--whisper-model", type=str, default=os.getenv("WHISPER_MODEL", "small"), help="Modèle faster-whisper.")
    parser.add_argument(
        "--whisper-precision",
        type=str,
        default=os.getenv("WHISPER_PRECISION", "int8"),
        help="compute_type faster-whisper (int8, float16, etc.).",
    )
    parser.add_argument("--device", type=str, default=os.getenv("WHISPER_DEVICE", "cpu"), help="cpu ou cuda.")
    parser.add_argument("--ollama-model", type=str, default=os.getenv("OLLAMA_MODEL", "llama3"), help="Nom du modèle Ollama.")
    parser.add_argument(
        "--ollama-host", type=str, default=os.getenv("OLLAMA_HOST", "http://localhost:11434"), help="URL Ollama."
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=os.getenv("SYSTEM_PROMPT", GLaDOS_SYSTEM_PROMPT),
        help="Prompt système pour forcer la personnalité.",
    )
    parser.add_argument(
        "--piper-model",
        type=str,
        default=os.getenv("PIPER_MODEL", "voices/en_US-glados-low.onnx"),
        help="Chemin vers le modèle Piper (.onnx).",
    )
    parser.add_argument(
        "--piper-config",
        type=str,
        default=os.getenv("PIPER_CONFIG", None),
        help="Chemin vers la config Piper (.json). Par défaut : <modèle>.json",
    )
    parser.add_argument("--speaker", type=int, default=None, help="Speaker id si le modèle en gère plusieurs.")
    parser.add_argument("--once", action="store_true", help="Ne faire qu'un tour puis quitter.")
    parser.add_argument("--no-play", action="store_true", help="Ne pas jouer le WAV (pour debug).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Chargement Whisper...")
    stt = WhisperSTT(
        model_name=args.whisper_model,
        device=args.device,
        compute_type=args.whisper_precision,
        language=args.language,
    )

    print("Initialisation Ollama...")
    llm = OllamaClient(
        model=args.ollama_model,
        host=args.ollama_host,
        system_prompt=args.system_prompt,
    )

    print("Initialisation Piper (voix GLaDOS)...")
    tts = PiperGladosTTS(
        model_path=Path(args.piper_model),
        config_path=Path(args.piper_config) if args.piper_config else None,
        speaker=args.speaker,
    )

    print("Prêt. Appuie sur Entrée pour parler (Ctrl+C pour quitter).")
    try:
        while True:
            input("→ ")
            audio = record_audio(seconds=args.listen_seconds, samplerate=args.samplerate)
            text = stt.transcribe(audio)
            if not text:
                print("… rien entendu.")
                if args.once:
                    break
                continue

            print(f"Vous: {text}")
            try:
                reply = llm.generate(text)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"[Erreur LLM] {exc}")
                if args.once:
                    break
                continue

            print(f"GLaDOS: {reply}")

            try:
                wav_path = tts.synthesize(reply)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"[Erreur TTS] {exc}")
                if args.once:
                    break
                continue

            if not args.no_play:
                play_wav(wav_path)

            if args.once:
                break

    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.")


if __name__ == "__main__":
    main()
