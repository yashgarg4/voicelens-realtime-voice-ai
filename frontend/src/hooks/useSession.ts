import { useCallback, useEffect, useState } from "react";
import type { Question, SessionRecord, WerResult } from "../types";

interface StartResponse {
  session_id: string;
  question: Question;
}

export interface SessionState {
  questions: Question[];
  selectedId: number | null;
  setSelectedId: (id: number) => void;
  selectedQuestion: Question | null;
  sessionId: string | null;
  loadError: string | null;
  history: SessionRecord[];
  wer: WerResult | null;
  /** POST /api/session/start for the selected question; returns ids for the WS. */
  begin: () => Promise<{ questionId: number; sessionId: string }>;
  /** POST /api/session/end for the active session. */
  finish: () => Promise<void>;
  /** Re-fetch the past-sessions list (call after a session ends). */
  refreshHistory: () => Promise<void>;
}

export function useSession(): SessionState {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [history, setHistory] = useState<SessionRecord[]>([]);
  const [wer, setWer] = useState<WerResult | null>(null);

  // Load the question bank once.
  useEffect(() => {
    let cancelled = false;
    fetch("/api/questions")
      .then((r) => {
        if (!r.ok) throw new Error(`/api/questions returned ${r.status}`);
        return r.json();
      })
      .then((data: { questions: Question[] }) => {
        if (cancelled) return;
        setQuestions(data.questions);
        if (data.questions.length > 0) setSelectedId(data.questions[0].id);
      })
      .catch((err) => {
        if (!cancelled)
          setLoadError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshHistory = useCallback(async () => {
    try {
      const r = await fetch("/api/sessions");
      if (!r.ok) return;
      const data: { sessions: SessionRecord[] } = await r.json();
      setHistory(data.sessions);
    } catch {
      /* non-fatal: dashboard just stays empty */
    }
  }, []);

  // Load history + WER comparison once on mount.
  useEffect(() => {
    refreshHistory();
    fetch("/api/wer")
      .then((r) => (r.ok ? r.json() : null))
      .then((data: WerResult | null) => data && setWer(data))
      .catch(() => {});
  }, [refreshHistory]);

  const selectedQuestion =
    questions.find((q) => q.id === selectedId) ?? null;

  const begin = useCallback(async () => {
    const res = await fetch("/api/session/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question_id: selectedId }),
    });
    if (!res.ok) throw new Error(`/api/session/start returned ${res.status}`);
    const data: StartResponse = await res.json();
    setSessionId(data.session_id);
    return { questionId: data.question.id, sessionId: data.session_id };
  }, [selectedId]);

  const finish = useCallback(async () => {
    if (!sessionId) return;
    try {
      await fetch("/api/session/end", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
    } finally {
      setSessionId(null);
    }
  }, [sessionId]);

  return {
    questions,
    selectedId,
    setSelectedId,
    selectedQuestion,
    sessionId,
    loadError,
    history,
    wer,
    begin,
    finish,
    refreshHistory,
  };
}
