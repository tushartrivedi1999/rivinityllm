import { DashboardShell } from "@/components/dashboard/shell"
import { Card } from "@/components/ui/card"

const nodes = [
  { vendor: "rivinity-cloud", node: "10.0.0.5", type: "A100", util: "73%", vram: "62/80 GB" },
  { vendor: "partner-gpu", node: "10.0.0.8", type: "H100", util: "41%", vram: "39/80 GB" }
]

export default function GpuMonitorPage() {
  return (
    <DashboardShell title="GPU Cluster Monitor">
      <div className="grid gap-4 md:grid-cols-2">
        {nodes.map((n) => (
          <Card key={n.node} className="space-y-1">
            <p className="font-medium">{n.vendor}</p>
            <p className="text-sm text-zinc-300">Node: {n.node}</p>
            <p className="text-sm text-zinc-300">GPU: {n.type}</p>
            <p className="text-sm text-zinc-300">Utilization: {n.util}</p>
            <p className="text-sm text-zinc-300">VRAM: {n.vram}</p>
          </Card>
        ))}
      </div>
    </DashboardShell>
  )
}
