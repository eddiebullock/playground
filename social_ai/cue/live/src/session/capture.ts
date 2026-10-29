export async function shareCall(): Promise<MediaStream> {
  return navigator.mediaDevices.getDisplayMedia({
    video: { frameRate: 15, width: { ideal: 1280 } },
    audio: true,
    // Chrome extras; ignored elsewhere.
    systemAudio: "include",
    selfBrowserSurface: "exclude",
    surfaceSwitching: "include",
  } as DisplayMediaStreamOptions);
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
