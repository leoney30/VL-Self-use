import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import whisperx
import torch
from typing import Dict
import librosa
from rich import print as rprint
import subprocess
import tempfile
import time

from core.config_utils import load_key
# 移除了 demucs_vl 的导入，直接从 whisperX_utils 导入 RAW_AUDIO_FILE
from core.all_whisper_methods.whisperX_utils import RAW_AUDIO_FILE, process_transcription, convert_video_to_audio, split_audio, save_results, save_language, compress_audio, CLEANED_CHUNKS_EXCEL_PATH
from core.step1_ytdlp import find_video_files

# 定义常量路径
MODEL_DIR = load_key("model_dir")  # 模型存储目录
WHISPER_FILE = "output/audio/for_whisper.mp3"  # Whisper输入音频文件路径
# 移除了 ENHANCED_VOCAL_PATH，因为它不再需要

def check_hf_mirror() -> str:
    """检查并返回最快的HuggingFace镜像站点"""
    mirrors = {
        'Official': 'huggingface.co',  # 官方镜像
        'Mirror': 'hf-mirror.com'      # 备用镜像
    }
    fastest_url = f"https://{mirrors['Official']}"
    best_time = float('inf')
    rprint("[cyan]🔍 Checking HuggingFace mirrors...[/cyan]")
    for name, domain in mirrors.items():
        try:
            if os.name == 'nt':
                cmd = ['ping', '-n', '1', '-w', '3000', domain]
            else:
                cmd = ['ping', '-c', '1', '-W', '3', domain]
            start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            response_time = time.time() - start
            if result.returncode == 0:
                if response_time < best_time:
                    best_time = response_time
                    fastest_url = f"https://{domain}"
                rprint(f"[green]✓ {name}:[/green] {response_time:.2f}s")
        except:
            rprint(f"[red]✗ {name}:[/red] Failed to connect")
    if best_time == float('inf'):
        rprint("[yellow]⚠️ All mirrors failed, using default[/yellow]")
    rprint(f"[cyan]🚀 Selected mirror:[/cyan] {fastest_url} ({best_time:.2f}s)")
    return fastest_url

def transcribe_audio(audio_file: str, start: float, end: float) -> Dict:
    """
    使用WhisperX转录音频片段
    
    参数:
        audio_file: 音频文件路径
        start: 开始时间（秒）
        end: 结束时间（秒）
    返回:
        包含转录结果的字典
    """
    os.environ['HF_ENDPOINT'] = check_hf_mirror()
    WHISPER_LANGUAGE = load_key("whisper.language")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rprint(f"🚀 Starting WhisperX using device: {device} ...")
    
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        batch_size = 16 if gpu_mem > 8 else 2
        compute_type = "float16" if torch.cuda.is_bf16_supported() else "int8"
        rprint(f"[cyan]🎮 GPU memory:[/cyan] {gpu_mem:.2f} GB, [cyan]📦 Batch size:[/cyan] {batch_size}, [cyan]⚙️ Compute type:[/cyan] {compute_type}")
    else:
        batch_size = 1
        compute_type = "int8"
        rprint(f"[cyan]📦 Batch size:[/cyan] {batch_size}, [cyan]⚙️ Compute type:[/cyan] {compute_type}")
    rprint(f"[green]▶️ Starting WhisperX for segment {start:.2f}s to {end:.2f}s...[/green]")
    
    try:
        if WHISPER_LANGUAGE == 'zh':
            model_name = "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper"
            local_model = os.path.join(MODEL_DIR, "Belle-whisper-large-v3-zh-punct-fasterwhisper")
        else:
            model_name = load_key("whisper.model")
            local_model = os.path.join(MODEL_DIR, model_name)
            
        if os.path.exists(local_model):
            rprint(f"[green]📥 Loading local WHISPER model:[/green] {local_model} ...")
            model_name = local_model
        else:
            rprint(f"[green]📥 Using WHISPER model from HuggingFace:[/green] {model_name} ...")

        vad_options = {"vad_onset": 0.500,"vad_offset": 0.363}
        asr_options = {"temperatures": [0],"initial_prompt": "",}
        whisper_language = None if 'auto' in WHISPER_LANGUAGE else WHISPER_LANGUAGE
        rprint("[bold yellow]**You can ignore warning of `Model was trained with torch 1.10.0+cu102, yours is 2.0.0+cu118...`**[/bold yellow]")
        model = whisperx.load_model(model_name, device, compute_type=compute_type, language=whisper_language, vad_options=vad_options, asr_options=asr_options, download_root=MODEL_DIR)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        ffmpeg_cmd = f'ffmpeg -y -i "{audio_file}" -ss {start} -t {end-start} -vn -ar 32000 -ac 1 "{temp_audio_path}"'
        subprocess.run(ffmpeg_cmd, shell=True, check=True, capture_output=True)
        
        try:
            audio_segment, sample_rate = librosa.load(temp_audio_path, sr=16000)
        finally:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

        rprint("[bold green]note: You will see Progress if working correctly[/bold green]")
        result = model.transcribe(audio_segment, batch_size=batch_size, print_progress=True)

        del model
        torch.cuda.empty_cache()

        save_language(result['language'])
        if result['language'] == 'zh' and WHISPER_LANGUAGE != 'zh':
            raise ValueError("Please specify the transcription language as zh and try again!")

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio_segment, device, return_char_alignments=False)

        torch.cuda.empty_cache()
        del model_a

        for segment in result['segments']:
            segment['start'] += start
            segment['end'] += start
            for word in segment['words']:
                if 'start' in word:
                    word['start'] += start
                if 'end' in word:
                    word['end'] += start
        return result
    except Exception as e:
        rprint(f"[red]WhisperX processing error:[/red] {e}")
        raise

# 移除了整个 enhance_vocals 函数，因为它只用于处理Demucs分离出的人声

def transcribe():
    """
    主转录流程函数，包含以下步骤：
    1. 视频转音频
    2. 音频压缩
    3. 音频分段
    4. 转录处理
    5. 结果合并与保存
    """
    if os.path.exists(CLEANED_CHUNKS_EXCEL_PATH):
        rprint("[yellow]⚠️ 转录结果已存在，跳过转录步骤。[/yellow]")
        return
    
    # 步骤0：视频转音频，生成 RAW_AUDIO_FILE
    video_file = find_video_files()
    convert_video_to_audio(video_file)

    # 移除了步骤1（Demucs人声分离）

    # 步骤1 (原步骤2)：直接使用原始音频进行压缩
    # 直接使用从视频中提取的原始音频文件
    whisper_audio = compress_audio(RAW_AUDIO_FILE, WHISPER_FILE)

    # 步骤2 (原步骤3)：分割音频
    segments = split_audio(whisper_audio)
    
    # 步骤3 (原步骤4)：转录音频片段
    all_results = []
    for start, end in segments:
        result = transcribe_audio(whisper_audio, start, end)
        all_results.append(result)
    
    # 步骤4 (原步骤5)：合并结果
    combined_result = {'segments': []}
    for result in all_results:
        combined_result['segments'].extend(result['segments'])
    
    # 步骤5 (原步骤6)：处理并保存多种格式的结果
    df = process_transcription(combined_result)
    save_results(df)
    
    output_txt = "output/transcript.txt"
    with open(output_txt, 'w', encoding='utf-8') as f:
        for segment in combined_result['segments']:
            f.write(segment['text'] + '\n')
    rprint(f"[green]✓ Saved transcript to:[/green] {output_txt}")
    
    output_srt = "output/transcript.srt"
    with open(output_srt, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(combined_result['segments'], 1):
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{segment['text']}\n\n")
    rprint(f"[green]✓ Saved SRT to:[/green] {output_srt}")

def format_timestamp(seconds: float) -> str:
    """
    将秒数转换为SRT格式的时间戳 (HH:MM:SS,mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

if __name__ == "__main__":
    transcribe()
