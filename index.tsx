
import React, { useState, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, LiveServerMessage, Modality, Blob } from '@google/genai';

// --- CÁC HÀM HỖ TRỢ XỬ LÝ ÂM THANH VÀ DỮ LIỆU ---
// Các hàm này giúp chuyển đổi dữ liệu giữa các định dạng khác nhau.

/**
 * Mã hóa một mảng byte (Uint8Array) thành chuỗi Base64.
 * Cần thiết để gửi dữ liệu âm thanh qua JSON.
 * @param bytes Dữ liệu nhị phân cần mã hóa.
 * @returns Chuỗi Base64.
 */
function encode(bytes: Uint8Array) {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/**
 * Giải mã một chuỗi Base64 trở lại thành mảng byte (Uint8Array).
 * Dùng để xử lý dữ liệu âm thanh nhận về từ API.
 * @param base64 Chuỗi Base64 cần giải mã.
 * @returns Mảng byte (Uint8Array).
 */
function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

/**
 * Giải mã dữ liệu âm thanh thô (raw audio) nhận từ API thành một AudioBuffer
 * mà trình duyệt có thể phát được.
 * @param data Dữ liệu âm thanh dạng Uint8Array.
 * @param ctx AudioContext để xử lý âm thanh.
 * @param sampleRate Tần số lấy mẫu của âm thanh (API của Gemini là 24000).
 * @param numChannels Số kênh âm thanh (thường là 1 cho mono).
 * @returns Một Promise chứa AudioBuffer có thể phát được.
 */
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

/**
 * Tạo một đối tượng Blob theo định dạng mà API yêu cầu từ dữ liệu âm thanh của microphone.
 * @param data Dữ liệu âm thanh từ microphone (Float32Array).
 * @returns Đối tượng Blob chứa dữ liệu âm thanh đã mã hóa Base64 và mimeType.
 */
function createBlob(data: Float32Array): Blob {
    const l = data.length;
    const int16 = new Int16Array(l);
    for (let i = 0; i < l; i++) {
      int16[i] = data[i] * 32768;
    }
    return {
      data: encode(new Uint8Array(int16.buffer)),
      // API yêu cầu định dạng âm thanh PCM với tần số 16000
      mimeType: 'audio/pcm;rate=16000',
    };
}


const App = () => {
  // --- QUẢN LÝ TRẠNG THÁI (STATE) CỦA ỨNG DỤNG ---
  // State dùng để lưu trữ những dữ liệu thay đổi và cần được render lại trên giao diện.
  const [isRecording, setIsRecording] = useState(false); // Trạng thái đang ghi âm (true) hay không (false).
  const [status, setStatus] = useState('Click the mic to start speaking'); // Tin nhắn trạng thái hiển thị cho người dùng.
  const [transcript, setTranscript] = useState<{ speaker: string, text: string }[]>([]); // Mảng lưu trữ toàn bộ cuộc hội thoại.
  const [corrections, setCorrections] = useState(''); // Chuỗi lưu trữ các góp ý, sửa lỗi từ AI.
  
  // --- REFS ĐỂ LƯU TRỮ CÁC ĐỐI TƯỢNG KHÔNG CẦN RENDER LẠI ---
  // Ref dùng để lưu các đối tượng phức tạp (như audio context, session) mà không làm component bị render lại mỗi khi chúng thay đổi.
  const transcriptEndRef = useRef<HTMLDivElement>(null); // Ref để trỏ tới cuối hộp thoại, giúp tự động cuộn xuống.

  const aiRef = useRef<GoogleGenAI | null>(null); // Lưu trữ đối tượng GoogleGenAI.
  const sessionPromiseRef = useRef<Promise<any> | null>(null); // Lưu trữ promise của phiên kết nối với Gemini Live API.
  const inputAudioContextRef = useRef<AudioContext | null>(null); // Context xử lý âm thanh đầu vào từ micro.
  const outputAudioContextRef = useRef<AudioContext | null>(null); // Context xử lý âm thanh đầu ra từ AI.
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null); // Node để xử lý các khối âm thanh từ micro.
  const userMediaStreamRef = useRef<MediaStream | null>(null); // Luồng dữ liệu từ microphone của người dùng.
  const mediaStreamSourceRef = useRef<MediaStreamAudioSourceNode | null>(null); // Nguồn âm thanh từ luồng micro.

  // Refs để lưu trữ dữ liệu tạm thời trong một phiên nói chuyện.
  const currentInputTranscriptionRef = useRef(''); // Lưu văn bản được chuyển đổi từ giọng nói của người dùng.
  const currentOutputTranscriptionRef = useRef(''); // Lưu văn bản được chuyển đổi từ giọng nói của AI.
  const nextStartTimeRef = useRef(0); // Dùng để hẹn giờ phát các đoạn âm thanh của AI một cách liền mạch.
  const audioSourcesRef = useRef(new Set<AudioBufferSourceNode>()); // Tập hợp các nguồn âm thanh đang chờ hoặc đang phát.

  // `useEffect` này sẽ được chạy mỗi khi `transcript` thay đổi.
  // Nó sẽ tự động cuộn xuống tin nhắn mới nhất.
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript]);

  /**
   * Gọi API Gemini để phân tích và đưa ra góp ý cho câu nói của người dùng.
   * @param userText Câu nói của người dùng cần phân tích.
   */
  const getCorrections = async (userText: string) => {
    if (!userText || !aiRef.current) return;
    setStatus('Analyzing your speech...');
    try {
      const response = await aiRef.current.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: `The user said: "${userText}"`,
        // `systemInstruction` chỉ dẫn cho AI cách hành xử.
        config: {
          systemInstruction: `You are an AI English tutor. The user is practicing their conversational English.
          Analyze the user's last sentence for grammatical errors, spelling mistakes, or potential mispronunciations (based on the text).
          - If there are errors, provide a corrected version and a very brief, simple explanation in a friendly and encouraging tone.
          - If the sentence is perfect, simply reply with "Great job! No corrections needed.".`,
        },
      });
      setCorrections(response.text); // Cập nhật state với góp ý từ AI.
      setStatus('Ready to listen');
    } catch (error) {
      console.error("Error getting corrections:", error);
      setCorrections("Sorry, I couldn't analyze the text right now.");
      setStatus('Ready to listen');
    }
  };

  /**
   * Dừng cuộc hội thoại và dọn dẹp tất cả tài nguyên đang sử dụng.
   */
  const stopConversation = async () => {
    setStatus('Stopping...');
    setIsRecording(false);
    
    // Đóng phiên kết nối với Gemini API.
    if (sessionPromiseRef.current) {
        const session = await sessionPromiseRef.current;
        session.close();
        sessionPromiseRef.current = null;
    }

    // Dừng luồng microphone.
    if (userMediaStreamRef.current) {
      userMediaStreamRef.current.getTracks().forEach(track => track.stop());
      userMediaStreamRef.current = null;
    }
    // Ngắt kết nối các node xử lý âm thanh.
    if (scriptProcessorRef.current) {
        scriptProcessorRef.current.disconnect();
        scriptProcessorRef.current = null;
    }
    if (mediaStreamSourceRef.current) {
        mediaStreamSourceRef.current.disconnect();
        mediaStreamSourceRef.current = null;
    }
    // Đóng các audio context để giải phóng bộ nhớ.
    if (inputAudioContextRef.current && inputAudioContextRef.current.state !== 'closed') {
      await inputAudioContextRef.current.close();
      inputAudioContextRef.current = null;
    }
    if (outputAudioContextRef.current && outputAudioContextRef.current.state !== 'closed') {
        await outputAudioContextRef.current.close();
        outputAudioContextRef.current = null;
    }

    // Dừng tất cả các âm thanh của AI đang chờ phát.
    audioSourcesRef.current.forEach(source => source.stop());
    audioSourcesRef.current.clear();
    nextStartTimeRef.current = 0;
    
    setStatus('Click the mic to start speaking');
  };

  /**
   * Bắt đầu cuộc hội thoại: xin quyền micro, kết nối tới Gemini Live API và thiết lập xử lý âm thanh.
   */
  const startConversation = async () => {
    // Reset trạng thái cho phiên mới.
    setTranscript([]);
    setCorrections('');
    setStatus('Connecting...');
    setIsRecording(true);

    try {
        // Khởi tạo đối tượng GoogleGenAI nếu chưa có.
        if (!aiRef.current) {
            // CẢNH BÁO BẢO MẬT: Không dùng cách này trong môi trường production.
            // API key của bạn sẽ bị lộ cho tất cả mọi người.
            // Hãy thay "YOUR_API_KEY_HERE" bằng API key thật của bạn để chạy thử.
            aiRef.current = new GoogleGenAI({ apiKey: "AIzaSyBZASF552XISdS99UaEuJRaIzHld6PARp8" });
        }
        
        // Tạo các AudioContext để xử lý âm thanh.
        // Tần số 16000Hz cho đầu vào (micro) và 24000Hz cho đầu ra (giọng nói AI) theo yêu cầu của API.
        inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
        outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
        
        // Xin quyền truy cập microphone từ người dùng.
        userMediaStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Bắt đầu kết nối tới Gemini Live API.
        sessionPromiseRef.current = aiRef.current.live.connect({
            model: 'gemini-2.5-flash-native-audio-preview-09-2025',
            // `callbacks` là các hàm sẽ được gọi khi có sự kiện xảy ra.
            callbacks: {
                // Được gọi khi kết nối thành công.
                onopen: () => {
                    setStatus('Listening... Say something!');
                    // Thiết lập pipeline xử lý âm thanh: micro -> source -> processor -> API.
                    mediaStreamSourceRef.current = inputAudioContextRef.current!.createMediaStreamSource(userMediaStreamRef.current!);
                    scriptProcessorRef.current = inputAudioContextRef.current!.createScriptProcessor(4096, 1, 1);
                    
                    // Hàm này được gọi liên tục mỗi khi có một khối dữ liệu âm thanh từ micro.
                    scriptProcessorRef.current.onaudioprocess = (audioProcessingEvent) => {
                        const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
                        const pcmBlob = createBlob(inputData); // Chuyển đổi dữ liệu sang định dạng API cần.
                        if (sessionPromiseRef.current) {
                           // Gửi dữ liệu âm thanh tới API.
                           sessionPromiseRef.current.then((session) => {
                             session.sendRealtimeInput({ media: pcmBlob });
                           });
                        }
                    };
                    
                    // Kết nối các node xử lý âm thanh lại với nhau.
                    mediaStreamSourceRef.current.connect(scriptProcessorRef.current);
                    scriptProcessorRef.current.connect(inputAudioContextRef.current!.destination);
                },
                // Được gọi mỗi khi nhận được tin nhắn từ server.
                onmessage: async (message: LiveServerMessage) => {
                  // Xử lý văn bản được chuyển đổi từ giọng nói (transcription).
                  if (message.serverContent?.outputTranscription) {
                      currentOutputTranscriptionRef.current += message.serverContent.outputTranscription.text;
                  } else if (message.serverContent?.inputTranscription) {
                      currentInputTranscriptionRef.current += message.serverContent.inputTranscription.text;
                  }

                  // `turnComplete` = true báo hiệu một lượt nói (của người dùng và AI) đã hoàn tất.
                  if (message.serverContent?.turnComplete) {
                      const userInput = currentInputTranscriptionRef.current.trim();
                      const aiOutput = currentOutputTranscriptionRef.current.trim();

                      // Cập nhật giao diện với nội dung hội thoại.
                      setTranscript(prev => {
                          const newTranscript = [...prev];
                          if (userInput) newTranscript.push({ speaker: 'You', text: userInput });
                          if (aiOutput) newTranscript.push({ speaker: 'AI', text: aiOutput });
                          return newTranscript;
                      });

                      // Nếu người dùng có nói, gọi hàm để lấy góp ý.
                      if (userInput) {
                          getCorrections(userInput);
                      }
                      
                      // Reset các biến tạm.
                      currentInputTranscriptionRef.current = '';
                      currentOutputTranscriptionRef.current = '';
                  }

                  // Xử lý dữ liệu âm thanh giọng nói của AI gửi về.
                  const audioData = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
                  if (audioData && outputAudioContextRef.current) {
                      // Lên lịch phát âm thanh để đảm bảo không bị ngắt quãng.
                      nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outputAudioContextRef.current.currentTime);
                      const audioBuffer = await decodeAudioData(decode(audioData), outputAudioContextRef.current, 24000, 1);
                      const source = outputAudioContextRef.current.createBufferSource();
                      source.buffer = audioBuffer;
                      source.connect(outputAudioContextRef.current.destination);
                      source.addEventListener('ended', () => audioSourcesRef.current.delete(source));
                      source.start(nextStartTimeRef.current);
                      // Cập nhật thời gian bắt đầu cho đoạn âm thanh tiếp theo.
                      nextStartTimeRef.current += audioBuffer.duration;
                      audioSourcesRef.current.add(source);
                  }
                },
                // Được gọi khi có lỗi xảy ra.
                onerror: (e: ErrorEvent) => {
                    console.error('Session error:', e);
                    setStatus('An error occurred. Please try again.');
                    stopConversation();
                },
                // Được gọi khi phiên kết nối đóng lại.
                onclose: () => {
                    console.log('Session closed.');
                },
            },
            // Cấu hình cho phiên kết nối.
            config: {
                responseModalities: [Modality.AUDIO], // Yêu cầu API trả về âm thanh.
                inputAudioTranscription: {}, // Bật tính năng chuyển đổi giọng nói người dùng thành văn bản.
                outputAudioTranscription: {}, // Bật tính năng chuyển đổi giọng nói AI thành văn bản.
                systemInstruction: 'You are a friendly English speaking partner. Keep your responses concise and natural, like a real friend. Engage in a casual conversation.'
            },
        });
        
        // Chờ cho đến khi kết nối được thiết lập hoàn toàn.
        await sessionPromiseRef.current;

    } catch (error) {
        console.error('Failed to start conversation:', error);
        setStatus('Could not start microphone. Please check permissions.');
        setIsRecording(false);
    }
  };

  /**
   * Xử lý sự kiện khi người dùng nhấn nút micro.
   * Bắt đầu hoặc dừng ghi âm tùy thuộc vào trạng thái hiện tại.
   */
  const handleToggleRecording = () => {
    if (isRecording) {
      stopConversation();
    } else {
      startConversation();
    }
  };

  // --- GIAO DIỆN NGƯỜI DÙNG (JSX) ---
  return (
    <div className="app-container">
      <header>
        <h1>AI English Speaking Tutor</h1>
        <p>{status}</p>
      </header>
      <main>
        <div className="transcript-container">
          {/* Hiển thị lịch sử hội thoại */}
          {transcript.map((entry, index) => (
            <div key={index} className={`message ${entry.speaker === 'You' ? 'user' : 'ai'}`}>
              <strong>{entry.speaker}</strong>
              {entry.text}
            </div>
          ))}
          {/* Hiển thị thẻ góp ý nếu có */}
          {corrections && (
            <div className="corrections-card">
              <h3>
                <span className="material-symbols-outlined">lightbulb</span>
                Feedback
              </h3>
              <p>{corrections}</p>
            </div>
          )}
          {/* Div trống để tự động cuộn xuống */}
          <div ref={transcriptEndRef} />
        </div>
      </main>
      <footer>
        <div className="controls">
          {/* Nút micro để bắt đầu/dừng ghi âm */}
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
