"use client"

import { useMemo, useState } from "react"

import { DashboardShell } from "@/components/dashboard/shell"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

type ChatMessage = {
  role: "user" | "assistant"
  content: string
}

const SAMPLE_RESPONSES = [
  "Sure — here is a concise summary: Rivinity Nexus orchestrates model lifecycle, datasets, training jobs, inference serving, and GPU scheduling through a unified control plane.",
  "Great prompt. For best quality, start with temperature 0.7 and max tokens 256, then tune based on determinism and output length needs.",
  "Streaming is active. I am returning this response token-by-token to simulate live model generation in a chat workflow."
]

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export default function InferencePage() {
  const [prompt, setPrompt] = useState("")
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(256)
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: "Hi! Ask me anything about your models and I will respond with streaming output."
    }
  ])
  const [isStreaming, setIsStreaming] = useState(false)

  const canSend = useMemo(() => prompt.trim().length > 0 && !isStreaming, [prompt, isStreaming])

  async function handleSend() {
    if (!canSend) return

    const userMessage: ChatMessage = { role: "user", content: prompt.trim() }
    const responseSeed = SAMPLE_RESPONSES[Math.floor(Math.random() * SAMPLE_RESPONSES.length)]
    const simulated = `${responseSeed} (temp=${temperature.toFixed(2)}, max_tokens=${maxTokens})`
    const tokens = simulated.split(" ")

    setMessages((prev) => [...prev, userMessage, { role: "assistant", content: "" }])
    setPrompt("")
    setIsStreaming(true)

    for (const token of tokens) {
      await sleep(28)
      setMessages((prev) => {
        const next = [...prev]
        const idx = next.length - 1
        next[idx] = {
          ...next[idx],
          content: next[idx].content ? `${next[idx].content} ${token}` : token
        }
        return next
      })
    }

    setIsStreaming(false)
  }

  return (
    <DashboardShell title="Inference Playground">
      <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
        <Card className="flex h-[72vh] flex-col">
          <div className="mb-3 flex items-center justify-between border-b border-border pb-3">
            <p className="font-medium">Chat Interface</p>
            <Badge>{isStreaming ? "Streaming" : "Idle"}</Badge>
          </div>

          <div className="flex-1 space-y-3 overflow-y-auto pr-1">
            {messages.map((message, idx) => (
              <div key={`${message.role}-${idx}`} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[85%] rounded-xl px-3 py-2 text-sm ${
                    message.role === "user" ? "bg-primary text-white" : "bg-zinc-900 text-zinc-100"
                  }`}
                >
                  {message.content || <span className="opacity-70">...</span>}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 space-y-2 border-t border-border pt-3">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="min-h-24 w-full rounded-md border border-border bg-zinc-950 p-3 text-sm"
              placeholder="Type your message..."
            />
            <div className="flex justify-end">
              <Button onClick={handleSend} disabled={!canSend}>
                {isStreaming ? "Generating..." : "Send"}
              </Button>
            </div>
          </div>
        </Card>

        <Card className="space-y-4">
          <p className="font-medium">Generation Controls</p>

          <div className="space-y-2">
            <label className="text-sm">Temperature: {temperature.toFixed(2)}</label>
            <input
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm">Max Tokens: {maxTokens}</label>
            <input
              type="range"
              min={16}
              max={2048}
              step={16}
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <p className="text-xs text-zinc-400">
            Responses currently stream in the UI to mimic token-by-token generation. Wire this to `/inference/stream` when auth/session integration is in place.
          </p>
        </Card>
      </div>
    </DashboardShell>
  )
}
