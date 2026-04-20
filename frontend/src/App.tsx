import {useState, useEffect, useRef} from 'react'
import ReactMarkdown from 'react-markdown'

type Message = { id: string; role: string; content: string; images?: string[] };
type Session = { id: string; title: string; messages: Message[] };

const generateId = () => Date.now().toString() + Math.random().toString(36).substring(2, 9);
const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = error => reject(error);
    });
};

export default function App() {
    const [sessions, setSessions] = useState<Session[]>([{
        id: generateId(),
        title: '新对话',
        messages: [{id: generateId(), role: 'ai', content: '你好！我是脐橙专家。可以上传多张图片，我会综合分析。'}]
    }]);
    const [activeSessionId, setActiveSessionId] = useState<string>(sessions[0].id);
    const [input, setInput] = useState('');
    const [loadingSessions, setLoadingSessions] = useState<string[]>([]);
    const [selectedImages, setSelectedImages] = useState<string[]>([]);
    const [sidebarWidth, setSidebarWidth] = useState(256);
    const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
    const [isResizingSidebar, setIsResizingSidebar] = useState(false);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const chatContainerRef = useRef<HTMLDivElement>(null);

    const currentSession = sessions.find(s => s.id === activeSessionId) || sessions[0];
    const isCurrentSessionLoading = loadingSessions.includes(activeSessionId);

    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [currentSession.messages]);

    useEffect(() => {
        if (!isResizingSidebar) return;

        const handleMouseMove = (event: MouseEvent) => {
            const nextWidth = Math.min(420, Math.max(220, event.clientX));
            setSidebarWidth(nextWidth);
        };

        const handleMouseUp = () => {
            setIsResizingSidebar(false);
        };

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [isResizingSidebar]);

    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || []);
        if (files.length + selectedImages.length > 4) {
            alert("最多上传 4 张图片哦！");
            return;
        }
        const newBase64s = await Promise.all(files.map(f => fileToBase64(f)));
        setSelectedImages(prev => [...prev, ...newBase64s]);
        if (e.target) e.target.value = '';
    };

    const removeImage = (index: number) => {
        setSelectedImages(prev => prev.filter((_, i) => i !== index));
    };

    const startSidebarResize = () => {
        if (!isSidebarCollapsed) {
            setIsResizingSidebar(true);
        }
    };

    const handleSend = async () => {
        if ((!input.trim() && selectedImages.length === 0) || isCurrentSessionLoading) return;
        const sessionIdForThisRequest = activeSessionId;

        const userMsg: Message = {id: generateId(), role: 'user', content: input, images: [...selectedImages]};
        const aiMsgId = generateId();
        const initialAiMsg: Message = {id: aiMsgId, role: 'ai', content: ''};

        setSessions(prev => prev.map(s => {
            if (s.id === sessionIdForThisRequest) {
                const newTitle = s.title === '新对话' ? (input.trim() ? input.slice(0, 8) : "多图分析") : s.title;
                return {...s, title: newTitle, messages: [...s.messages, userMsg, initialAiMsg]};
            }
            return s;
        }));

        const currentInput = input;
        const currentImages = [...selectedImages];
        setInput('');
        setSelectedImages([]);
        setLoadingSessions(prev => [...prev, sessionIdForThisRequest]);

        let accumulatedContent = '';

        try {
            const response = await fetch('http://localhost:8000/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    query: currentInput,
                    session_id: sessionIdForThisRequest,
                    images_base64: currentImages
                })
            });

            if (!response.body) throw new Error("无法读取数据流");

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let done = false;
            let buffer = '';

            while (!done) {
                const {value, done: readerDone} = await reader.read();
                done = readerDone;

                if (value) {
                    buffer += decoder.decode(value, {stream: true});
                    const parts = buffer.split('\n\n');
                    buffer = parts.pop() || '';

                    for (const part of parts) {
                        if (part.startsWith('data: ')) {
                            const dataStr = part.replace('data: ', '');
                            if (dataStr.trim() === '[DONE]') break;

                            try {
                                const parsed = JSON.parse(dataStr);
                                if (parsed.text) {
                                    accumulatedContent += parsed.text;
                                    setSessions(prev => prev.map(s => {
                                        if (s.id === sessionIdForThisRequest) {
                                            const updatedMessages = s.messages.map(m =>
                                                m.id === aiMsgId ? {...m, content: accumulatedContent} : m
                                            );
                                            return {...s, messages: updatedMessages};
                                        }
                                        return s;
                                    }));
                                }
                            } catch (e) {
                                console.warn("跳过不完整的数据块:", e);
                            }
                        }
                    }
                }
            }
        } catch (error) {
            setSessions(prev => prev.map(s => {
                if (s.id === sessionIdForThisRequest) {
                    const updatedMessages = s.messages.map(m =>
                        m.id === aiMsgId ? {...m, content: accumulatedContent + '\n\n❌ 网络异常，输出中断。'} : m
                    );
                    return {...s, messages: updatedMessages};
                }
                return s;
            }));
        } finally {
            setLoadingSessions(prev => prev.filter(id => id !== sessionIdForThisRequest));
        }
    };

    const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setInput(e.target.value);
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
        }
    };

    const cleanMarkdown = (text: string) => {
        if (!text) return '';
        return text
            .replace(/\n{3,}/g, '\n\n')
            .replace(/(#{1,6} .+)\n{2,}/g, '$1\n')
            .replace(/\n{2,}(#{1,6} )/g, '\n$1')
            .trim();
    };

    return (
        <div className="flex h-screen bg-gray-100 font-sans overflow-hidden text-gray-900">

            {/* 侧边栏 */}
            <div className="relative bg-gray-900 text-gray-300 flex flex-col z-20 shrink-0 transition-[width] duration-200" style={{width: isSidebarCollapsed ? 72 : sidebarWidth}}>
                <div className="text-xl font-bold mb-8 text-orange-500 flex items-center gap-2 px-2 mt-2">
                    🍊 脐橙专家系统
                </div>
                <button
                    onClick={() => {
                        const newId = generateId();
                        setSessions([{
                            id: newId,
                            title: '新对话',
                            messages: [{id: generateId(), role: 'ai', content: '你好！'}]
                        }, ...sessions.filter(s => s.messages.length > 1)]);
                        setActiveSessionId(newId);
                        setSelectedImages([]);
                    }}
                    className="w-full bg-orange-600 hover:bg-orange-500 text-white py-3 px-4 rounded-2xl transition-all mb-6 flex justify-center items-center gap-2 shadow-lg active:scale-95 shrink-0"
                >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"/>
                    </svg>
                    开启新咨询
                </button>
                <div className="flex-1 overflow-y-auto space-y-1">
                    {sessions.map(s => (
                        <div
                            key={s.id}
                            onClick={() => { setActiveSessionId(s.id); setSelectedImages([]); }}
                            className={`cursor-pointer p-3 rounded-xl truncate text-sm transition-all flex justify-between items-center ${
                                activeSessionId === s.id
                                    ? 'bg-gray-800 text-orange-400 font-medium border-l-4 border-orange-500'
                                    : 'hover:bg-gray-800'
                            }`}
                        >
                            <span className="truncate flex-1">{s.title}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* 主界面 */}
            <div className="flex-1 flex flex-col max-w-4xl mx-auto w-full h-screen bg-white shadow-sm">

                {/* 顶部栏 */}
                <div className="p-4 border-b border-gray-100 flex justify-between items-center bg-white/80 backdrop-blur z-10 shrink-0">
                    <span className="font-bold text-gray-800">{currentSession.title}</span>
                    <span className="text-[10px] bg-orange-50 text-orange-500 px-2.5 py-1 rounded-full font-bold border border-orange-100 uppercase tracking-wider">
                        SSE Stream Engine
                    </span>
                </div>

                {/* 消息列表 */}
                <div ref={chatContainerRef} className="flex-1 p-6 overflow-y-auto space-y-6 scroll-smooth">
                    {currentSession.messages.map((msg, index) => {
                        const isStreamingAiMessage =
                            msg.role === 'ai' &&
                            isCurrentSessionLoading &&
                            index === currentSession.messages.length - 1;
                        const hasAiContent = msg.role !== 'ai' || msg.content.trim() !== '';

                        return (
                        <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-[85%] rounded-3xl overflow-hidden ${
                                msg.role === 'user'
                                    ? 'bg-orange-500 text-white shadow-md'
                                    : 'bg-gray-100 text-gray-800'
                            }`}>
                                {/* 用户发送的图片 */}
                                {msg.images && msg.images.length > 0 && (
                                    <div className={`grid gap-1 p-2 ${msg.images.length === 1 ? 'grid-cols-1' : 'grid-cols-2'}`}>
                                        {msg.images.map((img, i) => (
                                            <img key={i} src={img}
                                                className="w-full aspect-square object-cover rounded-xl border border-black/5"/>
                                        ))}
                                    </div>
                                )}

                                <div className={`px-5 ${hasAiContent ? 'p-4' : 'pt-4 pb-3'}`}>
                                    {msg.role === 'ai' ? (
                                        <>
                                            {/* AI 的思考加载动画：放在气泡内部的顶部 */}
                                            {isStreamingAiMessage && (
                                                <div className="flex items-center gap-2 text-orange-500 text-[13px] font-medium animate-pulse mb-3 pb-3 border-b border-gray-200/60">
                                                    <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                                                    </svg>
                                                    <span>
                                                        {currentSession.messages[index - 1]?.images?.length
                                                            ? "专家正在深入分析图片特征..."
                                                            : "专家正在思考并检索知识库..."}
                                                    </span>
                                                </div>
                                            )}

                                            {/* AI 正文渲染 */}
                                            {msg.content.trim() !== '' && (
                                                <ReactMarkdown
                                                    className="
                                                        prose prose-sm prose-orange max-w-none
                                                        [&_h1]:text-base [&_h1]:font-semibold [&_h1]:mt-2 [&_h1]:mb-1
                                                        [&_h2]:text-base [&_h2]:font-semibold [&_h2]:mt-2 [&_h2]:mb-1
                                                        [&_h3]:text-sm  [&_h3]:font-semibold [&_h3]:mt-1.5 [&_h3]:mb-0.5
                                                        [&_p]:my-0.5 [&_p]:leading-relaxed
                                                        [&_ul]:my-1 [&_ul]:pl-4
                                                        [&_ol]:my-1 [&_ol]:pl-4
                                                        [&_li]:my-0 [&_li]:leading-relaxed
                                                        [&_li>p]:my-0 [&_li>p]:leading-relaxed
                                                        [&_blockquote]:my-1 [&_blockquote]:pl-3 [&_blockquote]:border-l-2 [&_blockquote]:border-orange-300
                                                        [&_pre]:my-1.5 [&_pre]:rounded-lg [&_pre]:text-xs
                                                        [&_code]:text-xs [&_code]:px-1 [&_code]:rounded
                                                        [&_table]:my-1.5 [&_table]:text-sm
                                                        [&_hr]:my-2
                                                        leading-relaxed
                                                    "
                                                >
                                                    {cleanMarkdown(msg.content)}
                                                </ReactMarkdown>
                                            )}
                                        </>
                                    ) : (
                                        <span className="leading-relaxed whitespace-pre-wrap">{msg.content}</span>
                                    )}
                                </div>
                            </div>
                        </div>
                    )})}
                </div>

                {/* 底部输入区 */}
                <div className="p-4 pb-6 pt-0 shrink-0 flex flex-col">
                    <div className="bg-[#f4f4f4] rounded-3xl border-2 border-transparent focus-within:bg-white focus-within:border-orange-200 focus-within:shadow-xl transition-all duration-300 relative">
                        {/* 多图预览序列 */}
                        {selectedImages.length > 0 && (
                            <div className="flex flex-wrap gap-3 px-4 pt-3 pb-1">
                                {selectedImages.map((img, idx) => (
                                    <div key={idx} className="relative group">
                                        <img src={img} className="w-16 h-16 rounded-xl object-cover shadow-sm transition-transform group-hover:scale-105"/>
                                        <button
                                            onClick={() => removeImage(idx)}
                                            className="absolute -top-2 -right-2 bg-gray-800 text-white rounded-full p-1 shadow-md hover:bg-red-500 transition-colors"
                                        >
                                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"/>
                                            </svg>
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}

                        <div className="flex items-end gap-2 p-2">
                            <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*" multiple className="hidden"/>

                            <button
                                onClick={() => fileInputRef.current?.click()}
                                className="p-2.5 mb-0.5 ml-1 text-gray-500 hover:text-orange-500 hover:bg-gray-200/50 rounded-full transition-all active:scale-95"
                            >
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                                        d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
                                </svg>
                            </button>

                            <textarea
                                ref={textareaRef}
                                value={input}
                                onChange={handleInputChange}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' && !e.shiftKey) {
                                        e.preventDefault();
                                        handleSend();
                                    }
                                }}
                                placeholder="描述病情，或上传病叶图片由专家分析..."
                                rows={1}
                                className="flex-1 px-1 py-3 bg-transparent focus:outline-none resize-none max-h-48 overflow-y-auto leading-relaxed text-gray-800 text-[15px] placeholder-gray-400"
                            />

                            <button
                                onClick={handleSend}
                                disabled={isCurrentSessionLoading || (!input.trim() && selectedImages.length === 0)}
                                className={`p-2.5 mb-1 mr-1 rounded-full flex items-center justify-center transition-all duration-200 ${
                                    (isCurrentSessionLoading || (!input.trim() && selectedImages.length === 0))
                                        ? 'bg-[#e5e5e5] text-gray-400 cursor-not-allowed'
                                        : 'bg-orange-500 text-white shadow-md shadow-orange-500/30 hover:bg-orange-600 active:scale-90 hover:-translate-y-0.5'
                                }`}
                            >
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2.5"
                                    strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                                    <path d="M12 19V5M5 12l7-7 7 7"/>
                                </svg>
                            </button>
                        </div>
                    </div>

                    <div className="text-[11px] text-gray-400 text-center mt-3 flex justify-center items-center gap-1">
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                        </svg>
                        内容由 AI 智能体生成，仅供参考，不作为最终农业诊断结果
                    </div>
                </div>
            </div>
        </div>
    );
}
