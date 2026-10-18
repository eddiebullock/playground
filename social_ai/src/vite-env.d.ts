/// <reference types="vite/client" />

export {};

declare global {
  interface Window {
    __start?: () => void;
    __seek?: (time: number) => void;
    __fieldSeek?: (time: number) => void;
  }
}
