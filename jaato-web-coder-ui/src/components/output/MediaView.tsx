/**
 * Plays / shows binary chunks a tool or the model produced for a
 * person (``ToolOutputEvent`` with ``mime_type`` + ``data_b64``).
 *
 * Images and container audio (wav/mp3/ogg…) go straight into the
 * browser's own elements.  Headerless PCM — what a speaking model
 * streams (``audio/pcm;rate=24000;encoding=signed-integer;bits=16``,
 * see ``STREAM_AUDIO_MIME``) — has nothing a media element can play,
 * so it is decoded by hand and queued through the Web Audio API,
 * chunk after chunk, with no gap between them.
 */
import { useEffect, useMemo, useRef } from "react";
import type { MediaItem } from "@/store/types";

function parsePcm(mime: string): { rate: number; bits: number; channels: number } | null {
  const [base, ...params] = mime.split(";").map((s) => s.trim());
  if (base !== "audio/pcm" && base !== "audio/l16" && base !== "audio/L16") return null;
  const p: Record<string, string> = {};
  for (const kv of params) {
    const [k, v] = kv.split("=");
    if (k && v) p[k.toLowerCase()] = v;
  }
  return { rate: Number(p.rate ?? 24000), bits: Number(p.bits ?? 16), channels: Number(p.channels ?? 1) };
}

function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

class PcmPlayer {
  private ctx: AudioContext | null = null;
  private nextAt = 0;
  private played = 0;

  feed(items: MediaItem[], rate: number, channels: number): void {
    for (; this.played < items.length; this.played++) {
      const item = items[this.played]!;
      const bytes = b64ToBytes(item.dataB64);
      const samples = new Int16Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 2));
      if (!samples.length) continue;
      this.ctx ??= new AudioContext({ sampleRate: rate });
      const frames = Math.floor(samples.length / channels);
      const buf = this.ctx.createBuffer(channels, frames, rate);
      for (let ch = 0; ch < channels; ch++) {
        const data = buf.getChannelData(ch);
        for (let i = 0; i < frames; i++) data[i] = samples[i * channels + ch]! / 32768;
      }
      const src = this.ctx.createBufferSource();
      src.buffer = buf;
      src.connect(this.ctx.destination);
      const startAt = Math.max(this.ctx.currentTime + 0.02, this.nextAt);
      src.start(startAt);
      this.nextAt = startAt + buf.duration;
    }
  }

  close(): void {
    this.ctx?.close().catch(() => undefined);
    this.ctx = null;
  }
}

function PcmStream({ items, mime }: { items: MediaItem[]; mime: string }) {
  const player = useRef<PcmPlayer | null>(null);
  const spec = useMemo(() => parsePcm(mime), [mime]);
  useEffect(() => {
    if (!spec) return;
    player.current ??= new PcmPlayer();
    player.current.feed(items, spec.rate, spec.channels);
  }, [items, spec]);
  useEffect(() => () => player.current?.close(), []);
  const seconds = spec ? items.reduce((n, it) => n + (it.dataB64.length * 3) / 4, 0) / (spec.rate * 2 * spec.channels) : 0;
  return (
    <div className="text-xs text-text-muted font-mono">
      🔊 pcm {spec?.rate ?? "?"} Hz · {seconds.toFixed(1)}s{items.at(-1)?.final ? "" : " · streaming"}
    </div>
  );
}

export function MediaView({ items }: { items: MediaItem[] }) {
  if (!items.length) return null;
  const byStream = new Map<string, MediaItem[]>();
  for (const it of items) {
    const key = it.streamId ?? `${it.mimeType}#single`;
    const list = byStream.get(key) ?? [];
    list.push(it);
    byStream.set(key, list);
  }
  return (
    <div className="flex flex-col gap-2 p-2">
      {[...byStream.entries()].map(([key, list]) => {
        const mime = list[0]!.mimeType;
        const base = mime.split(";")[0]!.trim().toLowerCase();
        if (base.startsWith("image/")) {
          const whole = list.map((x) => x.dataB64).join("");
          return <img key={key} alt="tool output" className="max-h-96 rounded-md border hairline" src={`data:${base};base64,${whole}`} />;
        }
        if (base === "audio/pcm" || base === "audio/l16") return <PcmStream key={key} items={list} mime={mime} />;
        if (base.startsWith("audio/")) {
          const whole = list.map((x) => x.dataB64).join("");
          return <audio key={key} controls preload="auto" src={`data:${base};base64,${whole}`} />;
        }
        if (base === "application/pdf") {
          const whole = list.map((x) => x.dataB64).join("");
          return <a key={key} className="text-primary underline text-xs" href={`data:${base};base64,${whole}`} target="_blank" rel="noreferrer">Open PDF attachment</a>;
        }
        return <div key={key} className="text-xs text-text-muted">[{mime}: {list.length} chunk(s), not renderable here]</div>;
      })}
    </div>
  );
}
