// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// Copyright 2026 Mozilla.ai
// Licensed under the Apache License, Version 2.0.

#pragma once

namespace ttsfile {

// Kept embedded so the push-to-talk prototype works from a single llamafile.
inline constexpr char kWebUi[] = R"HTML(<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llamafile speech tutor</title>
<style>
  body { max-width: 38rem; margin: 3rem auto; padding: 0 1rem; font: 1.1rem system-ui; }
  button, select { font: inherit; padding: .65rem 1rem; }
  #talk { min-width: 13rem; margin: 1rem 0; }
  #status { min-height: 3rem; }
</style>
<h1>Speech tutor</h1>
<p>Choose a language, then record a short turn. Stop recording to hear the reply.</p>
<label>Language
  <select id="language">
    <option value="">Detect automatically</option><option value="en">English</option>
    <option value="de">German</option><option value="es">Spanish</option>
    <option value="fr">French</option><option value="it">Italian</option>
    <option value="pt">Portuguese</option><option value="ru">Russian</option>
    <option value="zh">Chinese</option><option value="ja">Japanese</option>
    <option value="ko">Korean</option>
  </select>
</label>
<div><button id="talk">Start recording</button></div>
<p id="status" role="status">Ready.</p>
<audio id="reply" controls></audio>
<script>
const button = document.querySelector('#talk');
const status = document.querySelector('#status');
const player = document.querySelector('#reply');
let recorder, chunks = [], stream, busy = false;
const session = localStorage.getItem('ttsfile-session') ||
  (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random());
localStorage.setItem('ttsfile-session', session);

function wavBuffer(audio) {
  const channels = audio.numberOfChannels, frames = audio.length;
  const samples = new Float32Array(frames);
  for (let c = 0; c < channels; ++c) {
    const input = audio.getChannelData(c);
    for (let i = 0; i < frames; ++i) samples[i] += input[i] / channels;
  }
  const buffer = new ArrayBuffer(44 + frames * 2), view = new DataView(buffer);
  const str = (offset, value) => [...value].forEach((ch, i) => view.setUint8(offset + i, ch.charCodeAt(0)));
  str(0, 'RIFF'); view.setUint32(4, 36 + frames * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, audio.sampleRate, true);
  view.setUint32(28, audio.sampleRate * 2, true); view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); str(36, 'data'); view.setUint32(40, frames * 2, true);
  for (let i = 0; i < frames; ++i) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 32768 : s * 32767, true);
  }
  return buffer;
}

async function finishRecording() {
  stream.getTracks().forEach(track => track.stop());
  const encoded = new Blob(chunks, {type: recorder.mimeType});
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await encoded.arrayBuffer());
    const form = new FormData();
    form.append('file', new Blob([wavBuffer(decoded)], {type: 'audio/wav'}), 'recording.wav');
    form.append('session_id', session);
    const language = document.querySelector('#language').value;
    if (language) form.append('language', language);
    status.textContent = 'Thinking…';
    const response = await fetch('/v1/audio/speech-to-speech', {method: 'POST', body: form});
    if (!response.ok) {
      const error = await response.json().catch(() => null);
      throw new Error(error?.error?.message || `Request failed (${response.status})`);
    }
    player.src = URL.createObjectURL(await response.blob());
    try {
      await player.play();
      status.textContent = 'Reply ready. Record another turn when you are ready.';
    } catch (_) {
      status.textContent = 'Reply ready. Press Play to hear it, then record another turn.';
    }
  } finally {
    await context.close();
  }
}

button.addEventListener('click', async () => {
  if (busy) return;
  if (recorder?.state === 'recording') { recorder.stop(); return; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio: true});
    chunks = []; recorder = new MediaRecorder(stream);
    recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
    recorder.onstop = async () => {
      busy = true; button.disabled = true;
      try { await finishRecording(); }
      catch (error) { status.textContent = error.message; }
      finally { busy = false; button.disabled = false; button.textContent = 'Start recording'; }
    };
    recorder.start(); button.textContent = 'Stop and send';
    status.textContent = 'Recording…';
  } catch (error) { status.textContent = `Microphone unavailable: ${error.message}`; }
});
</script>)HTML";

}  // namespace ttsfile
