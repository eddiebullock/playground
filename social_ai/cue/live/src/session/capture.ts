export async function shareCall(preferBrowserTab = false): Promise<MediaStream> {
  return navigator.mediaDevices.getDisplayMedia({
    video: {
      frameRate: 15,
      width: { ideal: 1280 },
      ...(preferBrowserTab ? { displaySurface: "browser" } : {}),
    },
    audio: true,
    systemAudio: "include",
    selfBrowserSurface: "exclude",
    surfaceSwitching: "include",
    monitorTypeSurfaces: preferBrowserTab ? "exclude" : "include",
  } as DisplayMediaStreamOptions);
}

export async function shareMeetingTab(): Promise<MediaStream> {
  return shareCall(true);
}

export async function shareCamera(): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 640 }, frameRate: 15 },
    audio: false,
  });
}

export async function shareMic(): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({ audio: true, video: false });
}

export function stopStream(stream: MediaStream | null): void {
  stream?.getTracks().forEach((t) => t.stop());
}

export function captureElementStream(el: HTMLMediaElement): MediaStream | null {
  const media = el as HTMLMediaElement & {
    captureStream?: (fps?: number) => MediaStream;
    mozCaptureStream?: (fps?: number) => MediaStream;
  };
  const tap = media.captureStream ?? media.mozCaptureStream;
  if (!tap) return null;
  try {
    return tap.call(el, 15);
  } catch {
    return null;
  }
}
