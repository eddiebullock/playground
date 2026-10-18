type IconProps = {
  size?: number;
  color?: string;
};

function base(size: number) {
  return {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    "aria-hidden": true as const,
  };
}

export function ToneIcon({ size = 22, color = "currentColor" }: IconProps) {
  return (
    <svg {...base(size)}>
      <path
        d="M5 16.5 V8.2c0-1 .8-1.8 1.8-1.8h7.2c1 0 1.8.8 1.8 1.8v4.2c0 1-.8 1.8-1.8 1.8H9.2L5 16.5Z"
        stroke={color}
        strokeWidth="1.7"
        strokeLinejoin="round"
      />
      <path
        d="M17.2 9.2c.8.7 1.3 1.7 1.3 2.8s-.5 2.1-1.3 2.8"
        stroke={color}
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <path d="M8.2 11.2c.7-.5 1.6-.5 2.4 0 .7.4 1.6.4 2.3 0" stroke={color} strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

export function IdiomIcon({ size = 22, color = "currentColor" }: IconProps) {
  return (
    <svg {...base(size)}>
      <path
        d="M7 7.2c-1.6.9-2.6 2.2-2.6 3.6 0 1.5 1.2 2.6 2.8 2.6"
        stroke={color}
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <path d="M6.2 6.4v6.8" stroke={color} strokeWidth="1.7" strokeLinecap="round" />
      <path
        d="M15 7.2c-1.6.9-2.6 2.2-2.6 3.6 0 1.5 1.2 2.6 2.8 2.6"
        stroke={color}
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <path d="M14.2 6.4v6.8" stroke={color} strokeWidth="1.7" strokeLinecap="round" />
      <path d="M8 17.5h8" stroke={color} strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

export function TurnIcon({ size = 22, color = "currentColor" }: IconProps) {
  return (
    <svg {...base(size)}>
      <path
        d="M7 8.5h4.2c3.2 0 5.3 2 5.3 4.6 0 2.5-2 4.4-5 4.4H8"
        stroke={color}
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <path d="M9.2 15.2 7 17.5l2.2 2.2" stroke={color} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M8 6.2v4.2" stroke={color} strokeWidth="1.7" strokeLinecap="round" />
      <path d="M11.2 6.2v4.2" stroke={color} strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

export function TopicIcon({ size = 22, color = "currentColor" }: IconProps) {
  return (
    <svg {...base(size)}>
      <path d="M5 7.5h8.5" stroke={color} strokeWidth="1.7" strokeLinecap="round" />
      <path d="M5 12h6" stroke={color} strokeWidth="1.7" strokeLinecap="round" />
      <path
        d="M13 12h2.2c1.6 0 2.8 1.1 2.8 2.6V16"
        stroke={color}
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <path d="M16.2 14.6 18 16.4l1.8-1.8" stroke={color} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function UncertainIcon({ size = 22, color = "currentColor" }: IconProps) {
  return (
    <svg {...base(size)}>
      <path
        d="M8 8.2a4.2 4.2 0 0 1 7.4 2.4c0 2-1.6 2.8-2.6 3.5-.8.5-1.2 1-1.2 1.8"
        stroke={color}
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <circle cx="12" cy="18.2" r="0.9" fill={color} />
      <path
        d="M5.2 7.2a8 8 0 0 0 0 9.6M18.8 7.2a8 8 0 0 1 0 9.6"
        stroke={color}
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeDasharray="1.2 2.4"
      />
    </svg>
  );
}

export function AudioIcon({ size = 22, color = "currentColor" }: IconProps) {
  return (
    <svg {...base(size)}>
      <path d="M4.5 10h2.6l3.4-3.2v10.4L7.1 14H4.5V10Z" stroke={color} strokeWidth="1.7" strokeLinejoin="round" />
      <path d="M13.2 9.2a3.4 3.4 0 0 1 0 5.6" stroke={color} strokeWidth="1.7" strokeLinecap="round" />
      <path d="M15.6 7.2a6 6 0 0 1 0 9.6" stroke={color} strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

export function LockIcon({ size = 16, color = "currentColor" }: IconProps) {
  return (
    <svg {...base(size)}>
      <rect x="6" y="10.5" width="12" height="8" rx="1.6" stroke={color} strokeWidth="1.7" />
      <path d="M8.5 10.5V8.2a3.5 3.5 0 0 1 7 0v2.3" stroke={color} strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

export function QuestionIcon({ size = 18, color = "currentColor" }: IconProps) {
  return (
    <svg {...base(size)}>
      <circle cx="12" cy="12" r="8" stroke={color} strokeWidth="1.7" />
      <path
        d="M9.6 9.6a2.4 2.4 0 1 1 3.2 2.3c-.7.4-1.1.8-1.1 1.6"
        stroke={color}
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <circle cx="12" cy="16.2" r="0.8" fill={color} />
    </svg>
  );
}

export function PauseIcon({ size = 18, color = "currentColor" }: IconProps) {
  return (
    <svg {...base(size)}>
      <path d="M8 6.5v11" stroke={color} strokeWidth="1.8" strokeLinecap="round" />
      <path d="M16 6.5v11" stroke={color} strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

export function HarborMark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 28 28" aria-hidden="true">
      <path d="M4 18.5c0-6.2 4.4-11 10-11s10 4.8 10 11" fill="none" stroke="#8AA899" strokeWidth="1.8" />
      <path d="M8.2 18.5c0-3.4 2.6-6 5.8-6s5.8 2.6 5.8 6" fill="none" stroke="#F7F5F0" strokeWidth="1.8" />
      <path d="M4 18.5h20" stroke="#8AA899" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

export function MicIcon({ size = 22, color = "currentColor" }: IconProps) {
  return (
    <svg {...base(size)}>
      <rect x="9" y="4" width="6" height="10" rx="3" stroke={color} strokeWidth="1.7" />
      <path d="M7 11.5a5 5 0 0 0 10 0" stroke={color} strokeWidth="1.7" strokeLinecap="round" />
      <path d="M12 16.5V20" stroke={color} strokeWidth="1.7" strokeLinecap="round" />
      <path d="M8.5 20h7" stroke={color} strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

export function CameraIcon({ size = 22, color = "currentColor" }: IconProps) {
  return (
    <svg {...base(size)}>
      <rect x="3.5" y="7" width="12" height="10" rx="2" stroke={color} strokeWidth="1.7" />
      <path d="M15.5 11.2 20.2 8.8v6.4l-4.7-2.4v-1.6Z" stroke={color} strokeWidth="1.7" strokeLinejoin="round" />
    </svg>
  );
}

export function ShareIcon({ size = 22, color = "currentColor" }: IconProps) {
  return (
    <svg {...base(size)}>
      <rect x="5" y="8" width="10" height="10" rx="1.6" stroke={color} strokeWidth="1.7" />
      <path d="M9 8V5.8A1.8 1.8 0 0 1 10.8 4H18v8.2a1.8 1.8 0 0 1-1.8 1.8H15" stroke={color} strokeWidth="1.7" />
    </svg>
  );
}

export function LeaveIcon({ size = 22, color = "currentColor" }: IconProps) {
  return (
    <svg {...base(size)}>
      <path
        d="M8.2 7.2h5.2c.8 0 1.4.6 1.4 1.4v1.2"
        stroke={color}
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <path
        d="M14.8 14.2v1.2c0 .8-.6 1.4-1.4 1.4H8.2c-.8 0-1.4-.6-1.4-1.4V8.6c0-.8.6-1.4 1.4-1.4"
        stroke={color}
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <path d="M11 12h8" stroke={color} strokeWidth="1.7" strokeLinecap="round" />
      <path d="M16.6 9.6 19 12l-2.4 2.4" stroke={color} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function AdvertIcon({ size = 22, color = "currentColor" }: IconProps) {
  return (
    <svg {...base(size)}>
      <rect x="3.5" y="6" width="17" height="12" rx="2" stroke={color} strokeWidth="1.7" />
      <path d="M7.5 12h9" stroke={color} strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

export function CueKindIcon({ kind, size = 22, color = "currentColor" }: IconProps & { kind: string }) {
  if (kind === "tone") return <ToneIcon size={size} color={color} />;
  if (kind === "idiom") return <IdiomIcon size={size} color={color} />;
  if (kind === "turn") return <TurnIcon size={size} color={color} />;
  if (kind === "topic") return <TopicIcon size={size} color={color} />;
  if (kind === "advert") return <AdvertIcon size={size} color={color} />;
  return <UncertainIcon size={size} color={color} />;
}
