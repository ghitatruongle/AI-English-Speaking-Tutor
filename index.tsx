import React, { useState, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, LiveServerMessage, Modality, Blob } from '@google/genai';

// --- Base64 and Audio Helper Functions ---

function encode(bytes: Uint8Array) {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

function createBlob(data: Float32Array): Blob {
    const l = data.length;
    const int16 = new Int16Array(l);
    for (let i = 0; i < l; i++) {
      int16[i] = data[i] * 32768;
    }
    return {
      data: encode(new Uint8Array(int16.buffer)),
      mimeType: 'audio/pcm;rate=16000',
    };
}


const App = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState('Click the mic to start speaking');
  const [transcript, setTranscript] = useState<{ speaker: string, text: string }[]>([]);
  const [corrections, setCorrections] = useState('');
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  const aiRef = useRef<GoogleGenAI | null>(null);
  const sessionPromiseRef = useRef<Promise<any> | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const userMediaStreamRef = useRef<MediaStream | null>(null);
  const mediaStreamSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const currentInputTranscriptionRef = useRef('');
  const currentOutputTranscriptionRef = useRef('');
  const nextStartTimeRef = useRef(0);
  const audioSourcesRef = useRef(new Set<AudioBufferSourceNode>());

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript]);

  const getCorrections = async (userText: string) => {
    if (!userText || !aiRef.current) return;
    setStatus('Analyzing your speech...');
    try {
      const response = await aiRef.current.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: `The user said: "${userText}"`,
        config: {
          systemInstruction: `You are an AI English tutor. The user is practicing their conversational English.
          Analyze the user's last sentence for grammatical errors, spelling mistakes, or potential mispronunciations (based on the text).
          - If there are errors, provide a corrected version and a very brief, simple explanation in a friendly and encouraging tone.
          - If the sentence is perfect, simply reply with "Great job! No corrections needed.".`,
        },
      });
      setCorrections(response.text);
      setStatus('Ready to listen');
    } catch (error) {
      console.error("Error getting corrections:", error);
      setCorrections("Sorry, I couldn't analyze the text right now.");
      setStatus('Ready to listen');
    }
  };

  const stopConversation = async () => {
    setStatus('Stopping...');
    setIsRecording(false);
    
    if (sessionPromiseRef.current) {
        const session = await sessionPromiseRef.current;
        session.close();
        sessionPromiseRef.current = null;
    }

    if (userMediaStreamRef.current) {
      userMediaStreamRef.current.getTracks().forEach(track => track.stop());
      userMediaStreamRef.current = null;
    }
    if (scriptProcessorRef.current) {
        scriptProcessorRef.current.disconnect();
        scriptProcessorRef.current = null;
    }
    if (mediaStreamSourceRef.current) {
        mediaStreamSourceRef.current.disconnect();
        mediaStreamSourceRef.current = null;
    }
    if (inputAudioContextRef.current && inputAudioContextRef.current.state !== 'closed') {
      await inputAudioContextRef.current.close();
      inputAudioContextRef.current = null;
    }
    if (outputAudioContextRef.current && outputAudioContextRef.current.state !== 'closed') {
        await outputAudioContextRef.current.close();
        outputAudioContextRef.current = null;
    }

    // Clear pending audio
    audioSourcesRef.current.forEach(source => source.stop());
    audioSourcesRef.current.clear();
    nextStartTimeRef.current = 0;
    
    setStatus('Click the mic to start speaking');
  };

  const startConversation = async () => {
    setTranscript([]);
    setCorrections('');
    setStatus('Connecting...');
    setIsRecording(true);

    try {
        if (!aiRef.current) {
            aiRef.current = new GoogleGenAI({ apiKey: process.env.API_KEY });
        }
        
        // Fix: Cast window to any to support prefixed webkitAudioContext for older browsers
        inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
        outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
        
        userMediaStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        sessionPromiseRef.current = aiRef.current.live.connect({
            model: 'gemini-2.5-flash-native-audio-preview-09-2025',
            callbacks: {
                onopen: () => {
                    setStatus('Listening... Say something!');
                    mediaStreamSourceRef.current = inputAudioContextRef.current!.createMediaStreamSource(userMediaStreamRef.current!);
                    scriptProcessorRef.current = inputAudioContextRef.current!.createScriptProcessor(4096, 1, 1);
                    
                    scriptProcessorRef.current.onaudioprocess = (audioProcessingEvent) => {
                        const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
                        const pcmBlob = createBlob(inputData);
                        if (sessionPromiseRef.current) {
                           sessionPromiseRef.current.then((session) => {
                             session.sendRealtimeInput({ media: pcmBlob });
                           });
                        }
                    };
                    
                    mediaStreamSourceRef.current.connect(scriptProcessorRef.current);
                    scriptProcessorRef.current.connect(inputAudioContextRef.current!.destination);
                },
                onmessage: async (message: LiveServerMessage) => {
                  if (message.serverContent?.outputTranscription) {
                      currentOutputTranscriptionRef.current += message.serverContent.outputTranscription.text;
                  } else if (message.serverContent?.inputTranscription) {
                      currentInputTranscriptionRef.current += message.serverContent.inputTranscription.text;
                  }

                  if (message.serverContent?.turnComplete) {
                      const userInput = currentInputTranscriptionRef.current.trim();
                      const aiOutput = currentOutputTranscriptionRef.current.trim();

                      setTranscript(prev => {
                          const newTranscript = [...prev];
                          if (userInput) newTranscript.push({ speaker: 'You', text: userInput });
                          if (aiOutput) newTranscript.push({ speaker: 'AI', text: aiOutput });
                          return newTranscript;
                      });

                      if (userInput) {
                          getCorrections(userInput);
                      }
                      
                      currentInputTranscriptionRef.current = '';
                      currentOutputTranscriptionRef.current = '';
                  }

                  const audioData = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
                  if (audioData && outputAudioContextRef.current) {
                      nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outputAudioContextRef.current.currentTime);
                      const audioBuffer = await decodeAudioData(decode(audioData), outputAudioContextRef.current, 24000, 1);
                      const source = outputAudioContextRef.current.createBufferSource();
                      source.buffer = audioBuffer;
                      source.connect(outputAudioContextRef.current.destination);
                      source.addEventListener('ended', () => audioSourcesRef.current.delete(source));
                      source.start(nextStartTimeRef.current);
                      nextStartTimeRef.current += audioBuffer.duration;
                      audioSourcesRef.current.add(source);
                  }
                },
                onerror: (e: ErrorEvent) => {
                    console.error('Session error:', e);
                    setStatus('An error occurred. Please try again.');
                    stopConversation();
                },
                onclose: () => {
                    console.log('Session closed.');
                },
            },
            config: {
                responseModalities: [Modality.AUDIO],
                inputAudioTranscription: {},
                outputAudioTranscription: {},
                systemInstruction: 'You are a friendly English speaking partner. Keep your responses concise and natural, like a real friend. Engage in a casual conversation.'
            },
        });
        
        await sessionPromiseRef.current;

    } catch (error) {
        console.error('Failed to start conversation:', error);
        setStatus('Could not start microphone. Please check permissions.');
        setIsRecording(false);
    }
  };

  const handleToggleRecording = () => {
    if (isRecording) {
      stopConversation();
    } else {
      startConversation();
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>AI English Speaking Tutor</h1>
        <p>{status}</p>
      </header>
      <main>
        <div className="transcript-container">
          {transcript.map((entry, index) => (
            <div key={index} className={`message ${entry.speaker === 'You' ? 'user' : 'ai'}`}>
              <strong>{entry.speaker}</strong>
              {entry.text}
            </div>
          ))}
          {corrections && (
            <div className="corrections-card">
              <h3>
                <span className="material-symbols-outlined">lightbulb</span>
                Feedback
              </h3>
              <p>{corrections}</p>
            </div>
          )}
          <div ref={transcriptEndRef} />
        </div>
      </main>
      <footer>
        <div className="controls">
          <button
            className={`mic-button ${isRecording ? 'recording' : ''}`}
            onClick={handleToggleRecording}
            aria-label={isRecording ? 'Stop speaking' : 'Start speaking'}
          >
            <span className="material-symbols-outlined">
              {isRecording ? 'stop' : 'mic'}
            </span>
          </button>
        </div>
      </footer>
    </div>
  );
};

const container = document.getElementById('root');
const root = createRoot(container!);
root.render(<App />);
