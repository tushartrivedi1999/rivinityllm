import { cn } from "@/lib/utils"

export function Input({ className, ...props }: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn("w-full rounded-md border border-border bg-zinc-950 px-3 py-2 text-sm", className)} {...props} />
}
