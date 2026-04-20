import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'

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
const MAX_SIDEBAR_WIDTH = 380
const COLLAPSED_SIDEBAR_WIDTH = 72

const generateId = () => `${Date.now()}_${Math.random().toString(36).slice(2, 9)}`

const fileToBase64 = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = reject
  })

const createSession = (): Session => ({
  id: generateId(),
  title: NEW_CHAT_TITLE,
  messages: [{ id: generateId(), role: 'ai', content: WELCOME_TEXT }],
})

const normalizeAnswerText = (text: string) => {
  if (!text) return ''

  return text
    .replace(/\$\s*\\le\s*([^$]+?)\$/g, '≤$1')
    .replace(/\$\s*\\ge\s*([^$]+?)\$/g, '≥$1')
    .replace(/\^\s*\\circ/g, '°')
    .replace(/\\text\s*\{\s*([^}]+)\s*\}/g, '$1')
    .replace(/\$/g, '')
    .replace(/\\,/g, ' ')
    .replace(/\s+\)/g, ')')
    .replace(/\(\s+/g, '(')
    .replace(/[ \t]+\n/g, '\n')
    .trim()
}

export default function ChatApp() {
  const initialSessionRef = useRef<Session | null>(null)
  if (!initialSessionRef.current) {
    initialSessionRef.current = createSession()
  }

  const [sessions, setSessions] = useState<Session[]>([initialSessionRef.current])
  const [activeSessionId, setActiveSessionId] = useState(initialSessionRef.current.id)
  const [input, setInput] = useState('')
  const [selectedImages, setSelectedImages] = useState<string[]>([])
  const [loadingSessions, setLoadingSessions] = useState<string[]>([])
  const [searchMode, setSearchMode] = useState<'auto' | 'web' | 'local'>('auto')
  const [sidebarWidth, setSidebarWidth] = useState(280)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [resizingSidebar, setResizingSidebar] = useState(false)
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null)
  const [editingTitle, setEditingTitle] = useState('')
  // ✅ 新增：工具执行状态文字，单独显示，不混入正文
  const [statusText, setStatusText] = useState('')

  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)
  const renameInputRef = useRef<HTMLInputElement>(null)

  const currentSession = sessions.find((session) => session.id === activeSessionId) ?? sessions[0]
  const isCurrentSessionLoading = loadingSessions.includes(activeSessionId)
  const visibleSidebarWidth = sidebarCollapsed ? COLLAPSED_SIDEBAR_WIDTH : sidebarWidth

  useEffect(() => {
    if (!chatContainerRef.current) return
    chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
  }, [currentSession])

  useEffect(() => {
    if (!resizingSidebar) return

    const handleMouseMove = (event: MouseEvent) => {
      setSidebarWidth(Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, event.clientX)))
    }

    const handleMouseUp = () => setResizingSidebar(false)

    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)

    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [resizingSidebar])

  useEffect(() => {
    if (editingSessionId && renameInputRef.current) {
      renameInputRef.current.focus()
      renameInputRef.current.select()
    }
  }, [editingSessionId])

  // ✅ 通知后端清理某个 session 的记忆
  const clearBackendMemory = (sessionId: string) => {
    fetch('http://localhost:8888/api/clear_memory', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    }).catch(() => {
      // 网络失败不影响前端操作，静默忽略
    })
  }

  const handleNewChat = () => {
    const nextSession = createSession()

    setSessions((prev) => {
      // ✅ 找出即将被过滤掉的空会话，通知后端清理
      const emptySessions = prev.filter((session) => session.messages.length <= 1)
      emptySessions.forEach((session) => clearBackendMemory(session.id))

      return [nextSession, ...prev.filter((session) => session.messages.length > 1)]
    })

    setActiveSessionId(nextSession.id)
    setSelectedImages([])
    setEditingSessionId(null)
    setEditingTitle('')
  }

  const handleDeleteSession = (sessionIdToDelete: string) => {
    // ✅ 通知后端清理该 session 的记忆
    clearBackendMemory(sessionIdToDelete)

    setSessions((prev) => {
      if (prev.length === 1) {
        const nextSession = createSession()
        setActiveSessionId(nextSession.id)
        return [nextSession]
      }

      const nextSessions = prev.filter((session) => session.id !== sessionIdToDelete)

      if (activeSessionId === sessionIdToDelete) {
        setActiveSessionId(nextSessions[0].id)
      }

      return nextSessions
    })

    if (editingSessionId === sessionIdToDelete) {
      setEditingSessionId(null)
      setEditingTitle('')
    }
  }

  const startRenameSession = (sessionId: string, currentTitle: string) => {
    setEditingSessionId(sessionId)
    setEditingTitle(currentTitle)
  }

  const saveRenameSession = (sessionId: string) => {
    const nextTitle = editingTitle.trim() || NEW_CHAT_TITLE

    setSessions((prev) =>
      prev.map((session) =>
        session.id === sessionId
          ? {
              ...session,
              title: nextTitle,
            }
          : session,
      ),
    )

    setEditingSessionId(null)
    setEditingTitle('')
  }

  const cancelRenameSession = () => {
    setEditingSessionId(null)
    setEditingTitle('')
  }

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? [])
    if (files.length + selectedImages.length > 4) {
      alert('最多上传 4 张图片。')
      return
    }
    const images = await Promise.all(files.map(fileToBase64))
    setSelectedImages((prev) => [...prev, ...images])
    event.target.value = ''
  }

  const removeImage = (index: number) => {
    setSelectedImages((prev) => prev.filter((_, currentIndex) => currentIndex !== index))
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
    if ((!input.trim() && selectedImages.length === 0) || !currentSession || isCurrentSessionLoading) return

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
        return {
          ...session,
          title: session.title === NEW_CHAT_TITLE ? (input.trim() ? input.slice(0, 12) : '图片分析') : session.title,
          messages: [...session.messages, userMessage, aiMessage],
        }
      }),
    )

    const currentInput = input
    const currentImages = [...selectedImages]
    setInput('')
    setSelectedImages([])
    setStatusText('')  // ✅ 清空上一次的状态文字
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
    setLoadingSessions((prev) => (prev.includes(sessionId) ? prev : [...prev, sessionId]))

    let accumulatedContent = ''

    try {
      const response = await fetch('http://localhost:8888/api/chat', {
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
        const chunks = buffer.split('\n\n')
        buffer = chunks.pop() || ''

        for (const chunk of chunks) {
          if (!chunk.startsWith('data: ')) continue
          const payload = chunk.slice(6)
          if (payload.trim() === '[DONE]') {
            setStatusText('')  // ✅ 完成后清空状态文字
            continue
          }

          const parsed = JSON.parse(payload)
          if (parsed.error) {
            throw new Error(parsed.error)
          }
          // ✅ __STATUS__ 前缀的单独显示为状态栏，不混入正文
          if (parsed.text?.startsWith('__ASK_USER__:')) {
            // 把追问作为普通 AI 消息显示，和正常回答一样渲染
            const question = parsed.text.replace('__ASK_USER__:', '').trim()
            accumulatedContent = question
            updateAiMessage(sessionId, aiMessageId, accumulatedContent)
          } else if (parsed.text?.startsWith('__STATUS__:')) {
            setStatusText(parsed.text.replace('__STATUS__:', '').trim())
          } else if (parsed.text) {
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
      setStatusText('')  // ✅ 无论成功失败，结束后清空状态文字
    }
  }

  const handleInputChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(event.target.value)
    if (!textareaRef.current) return
    textareaRef.current.style.height = 'auto'
    textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
  }

  if (!currentSession) return null

  return (
    <div className="flex h-screen overflow-hidden bg-white text-gray-900">
      <aside
        className="relative flex shrink-0 flex-col bg-[#f7f7f3] text-gray-700 transition-[width] duration-200"
        style={{ width: visibleSidebarWidth }}
      >
        <div className={`flex items-center px-4 pt-5 ${sidebarCollapsed ? 'justify-center' : 'justify-between'}`}>
          {!sidebarCollapsed && <div className="truncate text-[18px] font-semibold text-gray-900">脐橙专家系统</div>}
          <button
            type="button"
            onClick={() => setSidebarCollapsed((prev) => !prev)}
            className="rounded-xl p-2 text-gray-500 transition-colors hover:bg-black/5 hover:text-gray-900"
            aria-label={sidebarCollapsed ? '展开侧边栏' : '收起侧边栏'}
            title={sidebarCollapsed ? '展开侧边栏' : '收起侧边栏'}
          >
            <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <rect x="3" y="4" width="18" height="16" rx="2" strokeWidth="2" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 4v16" />
              {sidebarCollapsed ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 9l3 3-3 3" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 9l-3 3 3 3" />
              )}
            </svg>
          </button>
        </div>

        <button
          type="button"
          onClick={handleNewChat}
          className={`mt-4 rounded-2xl bg-white text-gray-900 ring-1 ring-black/8 transition-all hover:bg-black/[0.03] active:scale-95 ${
            sidebarCollapsed
              ? 'mx-auto flex h-12 w-12 items-center justify-center px-0'
              : 'mx-4 flex items-center justify-center gap-2 px-4 py-3'
          }`}
        >
          <svg className="h-5 w-5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
          </svg>
          {!sidebarCollapsed && <span>开启新咨询</span>}
        </button>

        <div className={`mt-6 flex-1 space-y-1 overflow-y-auto ${sidebarCollapsed ? 'px-2' : 'px-4'}`}>
          {sessions.map((session) => {
            const active = session.id === activeSessionId
            const isEditing = editingSessionId === session.id

            return (
              <div
                key={session.id}
                className={`group flex w-full items-center rounded-xl text-sm transition-all ${
                  active ? 'bg-white text-gray-900 shadow-sm ring-1 ring-black/5' : 'hover:bg-black/[0.03]'
                } ${sidebarCollapsed ? 'justify-center px-0 py-3' : 'gap-2 p-3'}`}
                title={session.title}
              >
                <button
                  type="button"
                  onClick={() => {
                    if (isEditing) return
                    setActiveSessionId(session.id)
                    setSelectedImages([])
                  }}
                  className={`flex min-w-0 flex-1 items-center ${sidebarCollapsed ? 'justify-center' : 'gap-3'}`}
                >
                  <span className={`rounded-full ${active ? 'bg-gray-900' : 'bg-gray-400'} h-2 w-2 shrink-0`} />
                  {!sidebarCollapsed &&
                    (isEditing ? (
                      <input
                        ref={renameInputRef}
                        value={editingTitle}
                        onChange={(e) => setEditingTitle(e.target.value)}
                        onClick={(e) => e.stopPropagation()}
                        onBlur={() => saveRenameSession(session.id)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            e.preventDefault()
                            saveRenameSession(session.id)
                          }
                          if (e.key === 'Escape') {
                            e.preventDefault()
                            cancelRenameSession()
                          }
                        }}
                        className="min-w-0 flex-1 rounded-md border border-gray-200 bg-white px-2 py-1 text-sm outline-none focus:border-orange-300"
                      />
                    ) : (
                      <span className="truncate text-left">{session.title}</span>
                    ))}
                </button>

                {!sidebarCollapsed && !isEditing && (
                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        startRenameSession(session.id, session.title)
                      }}
                      className="rounded-lg p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-700"
                      aria-label="重命名对话"
                      title="重命名对话"
                    >
                      <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M15.232 5.232l3.536 3.536M9 13l6.768-6.768a2.5 2.5 0 113.536 3.536L12.536 16.536a4 4 0 01-1.414.95L7 19l1.514-4.122A4 4 0 019 13z"
                        />
                      </svg>
                    </button>

                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        handleDeleteSession(session.id)
                      }}
                      className="rounded-lg p-1.5 text-gray-400 hover:bg-red-50 hover:text-red-500"
                      aria-label="删除对话"
                      title="删除对话"
                    >
                      <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M19 7L5 7M10 11v6M14 11v6M9 7V5a1 1 0 011-1h4a1 1 0 011 1v2M6 7l1 12a1 1 0 001 1h8a1 1 0 001-1l1-12"
                        />
                      </svg>
                    </button>
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {!sidebarCollapsed && (
          <div
            onMouseDown={() => setResizingSidebar(true)}
            className="absolute right-0 top-0 h-full w-2 cursor-col-resize bg-transparent hover:bg-black/5 active:bg-black/10"
          />
        )}
      </aside>

      <main className="flex h-screen w-full flex-1 flex-col bg-white">
        <div className="shrink-0 bg-white/90 backdrop-blur">
          <div className="mx-auto flex w-full max-w-5xl items-center justify-between px-6 py-4">
            <span className="font-semibold text-gray-800">{currentSession.title}</span>
            <span className="rounded-full border border-orange-100 bg-orange-50 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider text-orange-500">
              SSE Stream Engine
            </span>
          </div>
        </div>

        <div ref={chatContainerRef} className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-6 overflow-y-auto px-6 py-6">
          {currentSession.messages.map((message, index) => {
            const isStreamingAiMessage =
              message.role === 'ai' &&
              isCurrentSessionLoading &&
              index === currentSession.messages.length - 1

            const hasAiContent = message.role !== 'ai' || message.content.trim() !== ''

            return (
              <div key={message.id} className={`flex w-full ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`w-fit max-w-[78%] overflow-hidden rounded-3xl ${
                    message.role === 'user' ? 'bg-[#f4f4f4] text-gray-900' : 'bg-[#f4f4f4] text-gray-800'
                  }`}
                >
                  {message.images && message.images.length > 0 && (
                    <div className={`grid gap-1 p-2 ${message.images.length === 1 ? 'grid-cols-1' : 'grid-cols-2'}`}>
                      {message.images.map((image, imageIndex) => (
                        <img
                          key={imageIndex}
                          src={image}
                          alt={`消息图片 ${imageIndex + 1}`}
                          className="aspect-square w-full rounded-xl border border-black/5 object-cover"
                        />
                      ))}
                    </div>
                  )}

                  <div className={`px-5 ${hasAiContent ? 'p-4' : 'pb-3 pt-4'}`}>
                    {message.role === 'ai' ? (
                      <>
                        {isStreamingAiMessage && (
                          <div className="mb-3 flex flex-col gap-1 border-b border-gray-200/60 pb-3">
                            {/* ✅ 通用加载动画 */}
                            <div className="flex items-center gap-2 text-[13px] font-medium text-orange-500 animate-pulse">
                              <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
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
                            {/* ✅ 工具执行状态单独显示，不混入正文 */}
                            {statusText !== '' && (
                              <div className="text-[12px] text-gray-400 pl-6">{statusText}</div>
                            )}
                          </div>
                        )}

                        {message.content.trim() !== '' && (
                          <ReactMarkdown
                            className="
                              prose prose-sm max-w-none
                              [&_h1]:text-base [&_h1]:font-semibold [&_h1]:mt-2 [&_h1]:mb-1
                              [&_h2]:text-base [&_h2]:font-semibold [&_h2]:mt-2 [&_h2]:mb-1
                              [&_h3]:text-sm [&_h3]:font-semibold [&_h3]:mt-1.5 [&_h3]:mb-0.5
                              [&_p]:my-1 [&_p]:leading-7
                              [&_ul]:my-1 [&_ul]:pl-5
                              [&_ol]:my-1 [&_ol]:pl-5
                              [&_li]:my-0.5 [&_li]:leading-7
                              [&_strong]:font-semibold
                              [&_code]:rounded [&_code]:bg-black/5 [&_code]:px-1
                              [&_pre]:rounded-lg [&_pre]:bg-black/5 [&_pre]:p-3
                            "
                          >
                            {normalizeAnswerText(message.content)}
                          </ReactMarkdown>
                        )}
                      </>
                    ) : (
                      <div className="whitespace-pre-wrap break-words text-[15px] leading-7">{message.content}</div>
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

            <div className="relative rounded-[28px] border border-black/6 bg-[#f4f4f4] transition-all duration-300 focus-within:border-orange-200 focus-within:bg-white focus-within:shadow-xl">
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
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <div className="flex items-end gap-2 p-2.5">
                <input ref={fileInputRef} type="file" onChange={handleFileChange} accept="image/*" multiple className="hidden" />

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
                      : 'bg-gray-900 text-white hover:-translate-y-0.5 hover:bg-black active:scale-90'
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
                    <path d="M12 19V5M5 12l7-7 7 7" />
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