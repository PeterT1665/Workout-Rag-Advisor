// Group a list into units: an item marked `superset` joins the item before it.
export function unitsOf<T extends { superset?: boolean }>(items: T[]): { start: number; items: T[] }[] {
  const out: { start: number; items: T[] }[] = []
  items.forEach((it, i) => {
    if (it.superset && out.length) out[out.length - 1].items.push(it)
    else out.push({ start: i, items: [it] })
  })
  return out
}

// Move unit `from` to position `to` (unit indexes) and flatten back to items.
export function moveUnit<T extends { superset?: boolean }>(items: T[], from: number, to: number): T[] {
  const units = unitsOf(items)
  const [u] = units.splice(from, 1)
  units.splice(to, 0, u)
  return units.flatMap((x) => x.items)
}
