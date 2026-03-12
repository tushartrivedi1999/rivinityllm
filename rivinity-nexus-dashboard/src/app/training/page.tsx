import { DashboardShell } from "@/components/dashboard/shell"
import { Card } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

const jobs = [
  { id: "job-102", model: "mistral-lite", loss: 0.21, progress: 78 },
  { id: "job-101", model: "llama-3-8b", loss: 0.34, progress: 42 }
]

export default function TrainingJobsPage() {
  return (
    <DashboardShell title="Training Jobs">
      <div className="space-y-4">
        {jobs.map((job) => (
          <Card key={job.id} className="space-y-2">
            <div className="flex justify-between text-sm"><span>{job.id}</span><span>{job.model}</span></div>
            <Progress value={job.progress} />
            <p className="text-sm text-zinc-300">Loss: {job.loss}</p>
          </Card>
        ))}
      </div>
    </DashboardShell>
  )
}
