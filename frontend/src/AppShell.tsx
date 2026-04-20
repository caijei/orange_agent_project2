import { useEffect, useRef, useState } from 'react'

type Message = {
  id: string
  role: 'user' | 'ai'
  content: string
  images?: string[]
}

type Session = {
  id: string
  title: string
  messages: Message[]
}

const NEW_CHAT_TITLE = '新对话'
const WELCOME_TEXT = '你好！我是脐橙专家。可以上传多张图片，我会综合分析。'
const MIN_SIDEBAR_WIDTH = 220
const MAX_SIDEBAR_WIDTH = 420
const COLLAPSED_SIDEBAR_WIDTH = 72

const generateId = () => `${Date.now()}_${Math.random().toString(36).slice(2, 9)}`

const fileToBase64 = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = reject
  })

const createInitialSession = (): Session => ({
  id: generateId(),
  title: NEW_CHAT_TITLE,
  messages: [{ id: generateId(), role: 'ai', content: WELCOME_TEXT }],
})

export default function AppShell() {
  const initialSessionRef = useRef<Session | null>(null)
  if (!initialSessionRef.current) {
    initialSessionRef.current = createInitialSession()
  }

  const [sessions, setSessions] = useState<Session[]>([initialSessionRef.current])
  const [activeSessionId, setActiveSessionId] = useState<string>(initialSessionRef.current.id)
  const [input, setInput] = useState('')
  const [selectedImages, setSelectedImages] = useState<string[]>([])
  const [loadingSessions, setLoadingSessions] = useState<string[]>([])
  const [searchMode, setSearchMode] = useState<'auto' | 'web' | 'local'>('auto')
  const [sidebarWidth, setSidebarWidth] = useState(280)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [isResizingSidebar, setIsResizingSidebar] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)

  const currentSession = sessions.find((session) => session.id === activeSessionId) ?? sessions[0]
  const isCurrentSessionLoading = loadingSessions.includes(activeSessionId)

  useEffect(() => {
    if (sessions.length > 0 && !sessions.some((session) => session.id === activeSessionId)) {
      setActiveSessionId(sessions[0].id)
    }
  }, [activeSessionId, sessions])

  useEffect(() => {
    if (!chatContainerRef.current) return
    chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
  }, [currentSession])

  useEffect(() => {
    if (!isResizingSidebar) return

    const handleMouseMove = (event: MouseEvent) => {
      const nextWidth = Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, event.clientX))
      setSidebarWidth(nextWidth)
    }

    const handleMouseUp = () => setIsResizingSidebar(false)

    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)

    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isResizingSidebar])

  const currentSidebarWidth = sidebarCollapsed ? COLLAPSED_SIDEBAR_WIDTH : sidebarWidth

  const handleNewSession = () => {
    const nextSession = createInitialSession()
    setSessions((prev) => [nextSession, ...prev.filter((session) => session.messages.length > 1)])
    setActiveSessionId(nextSession.id)
    setSelectedImages([])
  }

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? [])
    if (files.length + selectedImages.length > 4) {
      alert('最多上传 4 张图片。')
      return
    }

    const encodedFiles = await Promise.all(files.map(fileToBase64))
    setSelectedImages((prev) => [...prev, ...encodedFiles])
    event.target.value = ''
  }

  const removeImage = (index: number) => {
    setSelectedImages((prev) => prev.filter((_, current) => current !== index))
  }

  const updateAiMessage = (sessionId: string, messageId: string, content: string) => {
    setSessions((prev) =>
      prev.map((session) =>
        session.id === sessionId
          ? {
              ...session,
              messages: session.messages.map((message) =>
                message.id === messageId ? { ...message, content } : message,
              ),
            }
          : session,
      ),
    )
  }

  const handleSend = async () => {
    if ((!input.trim() && selectedImages.length === 0) || isCurrentSessionLoading || !currentSession) return

    const sessionId = currentSession.id
    const aiMessageId = generateId()
    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content: input,
      images: [...selectedImages],
    }
    const aiMessage: Message = { id: aiMessageId, role: 'ai', content: '' }

    setSessions((prev) =>
      prev.map((session) => {
        if (session.id !== sessionId) return session
        const nextTitle =
          session.title === NEW_CHAT_TITLE ? (input.trim() ? input.slice(0, 12) : '图片分析') : session.title
        return {
          ...session,
          title: nextTitle,
          messages: [...session.messages, userMessage, aiMessage],
        }
      }),
    )

    const currentInput = input
    const currentImages = [...selectedImages]
    setInput('')
    setSelectedImages([])
    setLoadingSessions((prev) => [...prev, sessionId])

    let accumulatedContent = ''

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: currentInput,
          session_id: sessionId,
          images_base64: currentImages,
          search_mode: searchMode,
        }),
      })

      if (!response.ok) {
        throw new Error(`请求失败：${response.status}`)
      }

      if (!response.body) {
        throw new Error('无法读取响应流。')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let done = false
      let buffer = ''

      while (!done) {
        const { value, done: readerDone } = await reader.read()
        done = readerDone

        if (!value) continue

        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop() || ''

        for (const part of parts) {
          if (!part.startsWith('data: ')) continue

          const payload = part.slice(6)
          if (payload.trim() === '[DONE]') continue

          const parsed = JSON.parse(payload)
          if (parsed.error) {
            throw new Error(parsed.error)
          }

          if (parsed.text) {
            accumulatedContent += parsed.text
            updateAiMessage(sessionId, aiMessageId, accumulatedContent)
          }
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : '网络异常，输出中断。'
      updateAiMessage(sessionId, aiMessageId, `${accumulatedContent}\n\n错误：${message}`.trim())
    } finally {
      setLoadingSessions((prev) => prev.filter((id) => id !== sessionId))
    }
  }

  const handleInputChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(event.target.value)
    if (!textareaRef.current) return
    textareaRef.current.style.height = 'auto'
    textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
  }

  const startSidebarResize = () => {
    if (!sidebarCollapsed) {
      setIsResizingSidebar(true)
    }
  }

  if (!currentSession) return null

  return (
    <div className="flex h-screen overflow-hidden bg-white text-gray-900">
      <aside
        className="relative flex shrink-0 flex-col bg-[#f7f7f3] text-gray-700 transition-[width] duration-200"
        style={{ width: currentSidebarWidth }}
      >
        <div className={`flex items-center pt-5 ${sidebarCollapsed ? 'justify-center px-2' : 'justify-between px-4'}`}>
          <div className={`flex items-center gap-2 text-xl font-semibold text-gray-900 ${sidebarCollapsed ? 'justify-center' : ''}`}>
            <span>橙</span>
            {!sidebarCollapsed && <span>脐橙专家系统</span>}
          </div>
          {!sidebarCollapsed && (
            <button
              onClick={() => setSidebarCollapsed(true)}
              className="rounded-xl p-2 text-gray-500 transition-colors hover:bg-black/5 hover:text-gray-900"
              aria-label="收起侧边栏"
            >
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" />
              </svg>
            </button>
          )}
        </div>

        {sidebarCollapsed && (
          <button
            onClick={() => setSidebarCollapsed(false)}
            className="mx-auto mt-4 rounded-xl p-2 text-gray-500 transition-colors hover:bg-black/5 hover:text-gray-900"
            aria-label="展开侧边栏"
          >
            <svg className="h-5 w-5 rotate-180" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" />
            </svg>
          </button>
        )}

        <button
          onClick={handleNewSession}
          className={`mt-4 rounded-2xl bg-white py-3 text-gray-900 ring-1 ring-black/8 transition-all hover:bg-black/[0.03] active:scale-95 ${
            sidebarCollapsed
              ? 'mx-auto flex h-12 w-12 items-center justify-center px-0'
              : 'mx-4 flex items-center justify-center gap-2 px-4'
          }`}
        >
          <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
          </svg>
          {!sidebarCollapsed && <span>开启新咨询</span>}
        </button>

        <div className={`mt-6 flex-1 space-y-1 overflow-y-auto ${sidebarCollapsed ? 'px-2' : 'px-4'}`}>
          {sessions.map((session) => {
            const active = session.id === activeSessionId
            return (
              <button
                key={session.id}
                onClick={() => {
                  setActiveSessionId(session.id)
                  setSelectedImages([])
                }}
                className={`flex w-full items-center rounded-xl text-sm transition-all ${
                  active ? 'bg-white text-gray-900 shadow-sm ring-1 ring-black/5' : 'hover:bg-black/[0.03]'
                } ${sidebarCollapsed ? 'justify-center px-0 py-3' : 'justify-between p-3'}`}
                title={session.title}
              >
                <span
                  className={`rounded-full ${
                    active ? 'bg-gray-900' : 'bg-gray-400'
                  } ${sidebarCollapsed ? 'h-2.5 w-2.5' : 'mr-3 h-2 w-2 shrink-0'}`}
                />
                {!sidebarCollapsed && <span className="truncate text-left">{session.title}</span>}
              </button>
            )
          })}
        </div>

        {!sidebarCollapsed && (
          <div
            onMouseDown={startSidebarResize}
            className="absolute right-0 top-0 h-full w-2 cursor-col-resize bg-transparent hover:bg-black/5 active:bg-black/10"
          />
        )}
      </aside>

      <main className="flex h-screen w-full flex-1 flex-col bg-white">
        <div className="shrink-0 bg-white/90 backdrop-blur">
          <div className="mx-auto flex w-full max-w-5xl items-center justify-between px-6 py-4">
            <span className="font-bold text-gray-800">{currentSession.title}</span>
            <span
                className="rounded-full border border-orange-100 bg-orange-50 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider text-orange-500">
              SSE Stream Engine
            </span>
          </div>
        </div>

        <div ref={chatContainerRef} className="mx-auto w-full max-w-5xl flex-1 space-y-6 overflow-y-auto px-6 py-6">
          {currentSession.messages.map((message, index) => {
            const isStreamingAiMessage =
                message.role === 'ai' &&
                isCurrentSessionLoading &&
                index === currentSession.messages.length - 1

            const hasAiContent = message.role !== 'ai' || message.content.trim() !== ''

            return (
                <div key={message.id}
                     className={`flex w-full ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div
                      className={`w-fit max-w-[78%] overflow-hidden rounded-3xl ${
                          message.role === 'user' ? 'bg-orange-500 text-white' : 'bg-gray-100 text-gray-800'
                      }`}
                  >
                    {message.images && message.images.length > 0 && (
                        <div
                            className={`grid gap-1 p-2 ${message.images.length === 1 ? 'grid-cols-1' : 'grid-cols-2'}`}>
                          {message.images.map((image, imageIndex) => (
                              <img
                                  key={imageIndex}
                                  src={image}
                                  className="aspect-square w-full rounded-xl border border-black/5 object-cover"
                              />
                          ))}
                        </div>
                    )}

                    <div className={`px-5 ${hasAiContent ? 'p-4' : 'pb-3 pt-4'}`}>
                      {message.role === 'ai' ? (
                          <>
                            {isStreamingAiMessage && (
                                <div
                                    className="mb-3 flex items-center gap-2 border-b border-gray-200/60 pb-3 text-[13px] font-medium text-orange-500 animate-pulse">
                                  <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor"
                                            strokeWidth="4"/>
                                    <path
                                        className="opacity-75"
                                        fill="currentColor"
                                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                    />
                                  </svg>
                                  <span>
                                    {currentSession.messages[index - 1]?.images?.length
                                        ? '专家正在深入分析图片特征...'
                                        : searchMode === 'web'
                                            ? '专家正在联网检索最新信息...'
                                            : searchMode === 'local'
                                                ? '专家正在检索本地知识库...'
                                                : '专家正在智能分析并选择检索方式...'}
                                  </span>
                                </div>
                            )}

                            {message.content.trim() !== '' && (
                                <div className="whitespace-pre-wrap break-words text-[15px] leading-relaxed">
                                {message.content}
                                </div>
                            )}
                          </>
                      ) : (
                          <div
                              className="whitespace-pre-wrap break-words text-[15px] leading-relaxed">{message.content}</div>
                      )}
                    </div>
                  </div>
                </div>
            )
          })}
        </div>

        <div className="shrink-0 px-6 pb-6 pt-2">
          <div className="mx-auto w-full max-w-4xl">
            <div className="mb-3 flex items-center gap-2 px-1">
              <span className="text-xs font-medium text-gray-500">检索模式</span>

              <button
                  type="button"
                  onClick={() => setSearchMode('auto')}
                  className={`rounded-full px-3 py-1 text-xs transition ${
                      searchMode === 'auto'
                          ? 'bg-orange-500 text-white'
                          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
              >
                智能
              </button>

              <button
                  type="button"
                  onClick={() => setSearchMode('web')}
                  className={`rounded-full px-3 py-1 text-xs transition ${
                      searchMode === 'web'
                          ? 'bg-orange-500 text-white'
                          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
              >
                联网
              </button>

              <button
                  type="button"
                  onClick={() => setSearchMode('local')}
                  className={`rounded-full px-3 py-1 text-xs transition ${
                      searchMode === 'local'
                          ? 'bg-orange-500 text-white'
                          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
              >
                知识库
              </button>
            </div>

            <div
                className="relative rounded-[28px] border border-black/6 bg-[#f4f4f4] transition-all duration-300 focus-within:border-orange-200 focus-within:bg-white focus-within:shadow-xl">
              {selectedImages.length > 0 && (
                  <div className="flex flex-wrap gap-3 px-4 pb-1 pt-3">
                    {selectedImages.map((image, index) => (
                        <div key={index} className="group relative">
                          <img
                              src={image}
                              alt={`待发送图片 ${index + 1}`}
                              className="h-16 w-16 rounded-xl object-cover shadow-sm transition-transform group-hover:scale-105"
                          />
                          <button
                              type="button"
                              onClick={() => removeImage(index)}
                              className="absolute -right-2 -top-2 rounded-full bg-gray-800 p-1 text-white shadow-md transition-colors hover:bg-red-500"
                          >
                            <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                                    d="M6 18L18 6M6 6l12 12"/>
                            </svg>
                          </button>
                        </div>
                    ))}
                  </div>
              )}

              <div className="flex items-end gap-2 p-2.5">
                <input
                    ref={fileInputRef}
                    type="file"
                    onChange={handleFileChange}
                    accept="image/*"
                    multiple
                    className="hidden"
                />

                <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="mb-0.5 ml-1 rounded-full p-2.5 text-gray-500 transition-all hover:bg-gray-200/50 hover:text-orange-500 active:scale-95"
                >
                  <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                </button>

                <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={handleInputChange}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' && !event.shiftKey) {
                        event.preventDefault()
                        handleSend()
                      }
                    }}
                    placeholder="描述病情，或上传病叶图片由专家分析..."
                    rows={1}
                    className="max-h-48 flex-1 resize-none overflow-y-auto bg-transparent px-1 py-3 text-[15px] leading-relaxed text-gray-800 placeholder-gray-400 focus:outline-none"
                />

                <button
                    type="button"
                    onClick={handleSend}
                    disabled={isCurrentSessionLoading || (!input.trim() && selectedImages.length === 0)}
                    className={`mb-1 mr-1 flex items-center justify-center rounded-full p-2.5 transition-all duration-200 ${
                        isCurrentSessionLoading || (!input.trim() && selectedImages.length === 0)
                            ? 'cursor-not-allowed bg-[#e5e5e5] text-gray-400'
                            : 'bg-orange-500 text-white shadow-md shadow-orange-500/30 hover:-translate-y-0.5 hover:bg-orange-600 active:scale-90'
                    }`}
                >
                  <svg
                      className="h-5 w-5"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      viewBox="0 0 24 24"
                  >
                    <path d="M12 19V5M5 12l7-7 7 7"/>
                  </svg>
                </button>
              </div>
            </div>
          </div>

                    <div className="mx-auto mt-3 flex w-full max-w-4xl items-center justify-center gap-1 text-center text-[11px] text-gray-400">
            <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <span>内容由 AI 智能体生成，仅供参考，不作为最终农业诊断结果。</span>
          </div>
        </div>
      </main>
    </div>
  )
}

