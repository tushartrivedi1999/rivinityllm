import Link from "next/link"

const links = [
  ["/models", "Model Registry"],
  ["/datasets", "Dataset Registry"],
  ["/training", "Training Jobs"],
  ["/inference", "Inference Playground"],
  ["/gpu", "GPU Cluster Monitor"]
]

export function Sidebar() {
  return (
    <aside className="w-64 border-r border-border p-4">
      <h2 className="mb-4 text-lg font-semibold">Rivinity Nexus</h2>
      <nav className="space-y-2">
        {links.map(([href, label]) => (
          <Link key={href} href={href} className="block rounded-md px-3 py-2 text-sm hover:bg-muted">
            {label}
          </Link>
        ))}
      </nav>
    </aside>
  )
}
