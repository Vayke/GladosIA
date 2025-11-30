# GLaDOS IA (local)

Pipeline local : micro → Whisper (faster-whisper) → LLM (Ollama) → TTS GLaDOS (Piper).

## Pré-requis
- Python 3.10+ conseillé.
- `pip install -r requirements.txt`
- **Ollama** : `curl https://ollama.ai/install.sh | sh`, puis `ollama pull llama3` (ou autre modèle local).
- **Piper** (synthèse vocale locale) : télécharge un binaire depuis https://github.com/rhasspy/piper/releases, ajoute-le à ton `PATH`.
- **Voix GLaDOS Piper** : télécharge le modèle et sa config, par exemple :
  ```bash
  mkdir -p voices
  curl -L -o voices/en_US-glados-low.onnx \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en_US/glados_low/en_US-glados-low.onnx
  curl -L -o voices/en_US-glados-low.onnx.json \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en_US/glados_low/en_US-glados-low.onnx.json
  ```
  Si ces URLs changent, prends une autre voix Piper au format `.onnx` + `.onnx.json`.

## Démarrage rapide
```bash
python -m glados_ia.main \
  --listen-seconds 3 \
  --whisper-model small \
  --ollama-model llama3 \
  --piper-model voices/en_US-glados-low.onnx
```
Appuie sur Entrée, parle, attends la réponse audio.

## Paramètres utiles
- `--listen-seconds` : durée d’enregistrement par tour (par défaut 3s).
- `--language` : forcer la langue Whisper (`en`, `fr`, etc.), sinon auto.
- `--device` et `--whisper-precision` : `cpu`/`cuda`, `int8`/`float16` pour équilibrer perf/latence.
- `--system-prompt` : prompt système pour la personnalité (par défaut GLaDOS).
- `--once` : un seul tour puis quitte.
- `--no-play` : ne joue pas le WAV (debug).

## Notes
- Les modèles Whisper se téléchargent au premier lancement (prévoir quelques centaines de Mo selon la taille).
- Assure-toi que `piper` et la voix sont bien accessibles, sinon la synthèse échouera.
- Pour une latence plus faible : modèle Whisper plus petit (`base`/`tiny`), réduire `--listen-seconds`, et garde Ollama/Piper préchargés.
