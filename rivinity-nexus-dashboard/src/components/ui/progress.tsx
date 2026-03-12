"use client"

import * as ProgressPrimitive from "@radix-ui/react-progress"

export function Progress({ value }: { value: number }) {
  return (
    <ProgressPrimitive.Root className="relative h-2 w-full overflow-hidden rounded-full bg-muted" value={value}>
      <ProgressPrimitive.Indicator
        className="h-full bg-primary transition-all"
        style={{ transform: `translateX(-${100 - (value || 0)}%)` }}
      />
    </ProgressPrimitive.Root>
  )
}
