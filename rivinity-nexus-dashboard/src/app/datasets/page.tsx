import { DashboardShell } from "@/components/dashboard/shell"
import { Card } from "@/components/ui/card"

const datasets = [
  { name: "support-conversations", format: "jsonl", shards: 16, status: "preprocessed" },
  { name: "product-catalog", format: "parquet", shards: 8, status: "uploaded" }
]

export default function DatasetsPage() {
  return (
    <DashboardShell title="Dataset Registry">
      <div className="grid gap-4 md:grid-cols-2">
        {datasets.map((d) => (
          <Card key={d.name} className="space-y-2">
            <p className="text-lg font-medium">{d.name}</p>
            <p className="text-sm text-zinc-300">Format: {d.format}</p>
            <p className="text-sm text-zinc-300">Shards: {d.shards}</p>
            <p className="text-sm text-zinc-300">Status: {d.status}</p>
          </Card>
        ))}
      </div>
    </DashboardShell>
  )
}
