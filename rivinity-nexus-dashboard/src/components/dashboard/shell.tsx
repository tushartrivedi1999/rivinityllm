import { Sidebar } from "@/components/dashboard/sidebar"

export function DashboardShell({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <main className="flex min-h-screen">
      <Sidebar />
      <section className="flex-1 p-6">
        <h1 className="mb-6 text-2xl font-semibold">{title}</h1>
        {children}
      </section>
    </main>
  )
}
