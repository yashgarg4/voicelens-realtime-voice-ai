// Typed contract for messages the backend sends over /ws/session.
// The browser only ever sends binary PCM frames (mic audio) plus optional
// JSON control frames, so we only need to model the *incoming* direction here.

export interface Question {
  id: number;
  category: string;
  difficulty: string;
  question: string;
  focus: string[];
}

export interface FeedbackScores {
  content: number; // 0-10
  depth: number; // 0-10
  structure: number; // 0-10
  overall: number;
  summary: string;
  strengths?: string;
  improvements?: string;
}

// A persisted session row from GET /api/sessions.
export interface SessionRecord {
  id: string;
  question_id: number | null;
  category: string | null;
  question: string | null;
  started_at: string | null;
  ended_at: string | null;
  content: number | null;
  depth: number | null;
  structure: number | null;
  overall: number | null;
  summary: string | null;
}

// GET /api/wer — base vs fine-tuned Whisper comparison from Phase 3.
export interface WerResult {
  available: boolean;
  base_wer?: number;
  finetuned_wer?: number;
  delta?: number;
  relative?: number;
  model?: string;
  dataset?: string;
  samples?: number;
}

export interface StatusMessage {
  type: "status";
  status: string;
  question?: Question;
}

export interface FeedbackMessage {
  type: "feedback";
  scores: FeedbackScores;
}

// The candidate's answer re-transcribed by the fine-tuned Whisper (B2 layer).
export interface FinetunedTranscriptMessage {
  type: "finetuned_transcript";
  text: string;
  using_adapter: boolean;
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export interface AudioMessage {
  type: "audio";
  data: string; // base64-encoded 24 kHz PCM16
}

export interface TextMessage {
  type: "text";
  text: string;
}

export interface TranscriptMessage {
  type: "input_transcript" | "output_transcript";
  text: string;
}

export interface SignalMessage {
  type: "turn_complete" | "interrupted" | "generation_complete" | "setup_complete";
}

export type ServerMessage =
  | StatusMessage
  | ErrorMessage
  | AudioMessage
  | TextMessage
  | TranscriptMessage
  | SignalMessage
  | FeedbackMessage
  | FinetunedTranscriptMessage;

export type ConnectionStatus =
  | "idle"
  | "connecting"
  | "reconnecting" // socket dropped; retrying with backoff
  | "speaking" // the coach is talking — wait your turn
  | "listening" // your turn to speak
  | "error"
  | "closed";

// One completed round of the conversation, stored in chat history.
export interface Exchange {
  id: number;
  user: string; // what you said (may be empty for the opening greeting)
  coach: string; // what the coach said
  at: number; // epoch ms
}
