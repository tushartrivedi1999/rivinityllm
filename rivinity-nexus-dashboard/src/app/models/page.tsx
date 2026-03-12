import { DashboardShell } from "@/components/dashboard/shell"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"

const models = [
  { name: "mistral-lite", version: "v2", status: "ready", owner: "team-a" },
  { name: "llama-3-8b", version: "v1", status: "training", owner: "team-ml" }
]

export default function ModelsPage() {
  return (
    <DashboardShell title="Model Registry">
      <Card>
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-border">
              <th className="py-2">Model</th><th>Version</th><th>Status</th><th>Owner</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={`${m.name}-${m.version}`} className="border-b border-border/50">
                <td className="py-3">{m.name}</td>
                <td>{m.version}</td>
                <td><Badge>{m.status}</Badge></td>
                <td>{m.owner}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </DashboardShell>
  )
}
