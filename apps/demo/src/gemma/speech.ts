// Voice input via Chrome's Web Speech API (on-device recognition when the language pack is available,
// Chrome 139+). Pattern from the LiteRT samples SAM 2 web demo (speech.ts, Apache-2.0), trimmed.

interface RecognitionResult { isFinal: boolean; 0: { transcript: string } }
interface RecognitionEvent { resultIndex: number; results: ArrayLike<RecognitionResult> }
interface Recognition {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  processLocally?: boolean;
  onresult: ((ev: RecognitionEvent) => void) | null;
  onerror: ((ev: { error: string }) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
}
type Availability = 'available' | 'downloadable' | 'downloading' | 'unavailable';
interface RecognitionCtor {
  new (): Recognition;
  available?(o: { langs: string[]; processLocally: boolean }): Promise<Availability>;
  install?(o: { langs: string[]; processLocally: boolean }): Promise<boolean>;
}
const Ctor = (window as unknown as { SpeechRecognition?: RecognitionCtor }).SpeechRecognition ??
  (window as unknown as { webkitSpeechRecognition?: RecognitionCtor }).webkitSpeechRecognition;

export const speechSupported = () => !!Ctor;

let active: Recognition | null = null;
export const listening = () => !!active;
export function stopListening(): void { active?.stop(); }

/** Listen for one utterance; `onText(text, final)` streams the transcript. */
export async function listen(lang: string, onText: (text: string, final: boolean) => void, onEnd: (err?: string) => void): Promise<'on-device' | 'Chrome server'> {
  if (!Ctor) throw new Error('speech recognition not supported in this browser');
  let local = false;
  try {
    const o = { langs: [lang], processLocally: true };
    const a = await Ctor.available?.(o);
    local = a === 'available' || ((a === 'downloadable' || a === 'downloading') && !!(await Ctor.install?.(o)));
  } catch { /* fall back to server recognition */ }
  const r = new Ctor();
  r.lang = lang;
  r.interimResults = true;
  r.continuous = false;
  if (local) r.processLocally = true;
  let text = '';
  r.onresult = (ev) => {
    let interim = '';
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const res = ev.results[i]!;
      if (res.isFinal) text += res[0].transcript;
      else interim += res[0].transcript;
    }
    onText((text + interim).trim(), false);
  };
  r.onerror = (ev) => { active = null; onEnd(ev.error); };
  r.onend = () => {
    active = null;
    if (text.trim()) onText(text.trim(), true);
    onEnd();
  };
  active = r;
  r.start();
  return local ? 'on-device' : 'Chrome server';
}
