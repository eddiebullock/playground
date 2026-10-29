import { useEffect, useRef } from "react";
import type { FaceBox } from "../session/vision.ts";

type Props = {
  video: { current: HTMLVideoElement | null };
  faces: FaceBox[];
  blur: number;
  active: boolean;
};

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  const rad = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

function contain(cw: number, ch: number, vw: number, vh: number) {
  const scale = Math.min(cw / vw, ch / vh);
  const w = vw * scale;
  const h = vh * scale;
  return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
}

export function FaceBlur({ video, faces, blur, active }: Props) {
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const facesRef = useRef(faces);
  const blurRef = useRef(blur);
  facesRef.current = faces;
  blurRef.current = blur;

  useEffect(() => {
    if (!active || blur <= 0) return;
    const cvs = canvas.current;
    const el = video.current;
    if (!cvs || !el) return;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const vw = el.videoWidth;
      const vh = el.videoHeight;
      if (!vw || !vh) return;
      const rect = cvs.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const cw = Math.max(1, Math.round(rect.width * dpr));
      const ch = Math.max(1, Math.round(rect.height * dpr));
      if (cvs.width !== cw || cvs.height !== ch) {
        cvs.width = cw;
        cvs.height = ch;
      }
      const box = contain(cw, ch, vw, vh);
      ctx.clearRect(0, 0, cw, ch);
      ctx.filter = `blur(${blurRef.current * dpr}px)`;
      ctx.drawImage(el, box.x, box.y, box.w, box.h);
      ctx.filter = "none";
      const face = facesRef.current[0];
      if (!face) return;
      const pad = 0.2;
      const fx = box.x + Math.max(0, face.x - face.w * pad) * box.w;
      const fy = box.y + Math.max(0, face.y - face.h * pad) * box.h;
      const fw = Math.min(box.w, face.w * (1 + pad * 2) * box.w);
      const fh = Math.min(box.h, face.h * (1 + pad * 2) * box.h);
      const r = Math.min(24 * dpr, fw / 4, fh / 4);
      ctx.save();
      roundRect(ctx, fx, fy, fw, fh, r);
      ctx.clip();
      ctx.drawImage(el, box.x, box.y, box.w, box.h);
      ctx.restore();
      ctx.strokeStyle = "rgba(240, 160, 102, 0.9)";
      ctx.lineWidth = 2 * dpr;
      roundRect(ctx, fx, fy, fw, fh, r);
      ctx.stroke();
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [active, blur, video]);

  if (!active || blur <= 0) return null;
  return <canvas ref={canvas} className="stage-blur" aria-hidden="true" />;
}
