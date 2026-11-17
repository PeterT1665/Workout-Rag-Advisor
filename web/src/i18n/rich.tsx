import { Fragment, type ReactNode } from 'react'
import { t } from '.'

// A translated line with bold words, a highlighted part and inline pieces, kept where the translation puts them:
// tr('Tap {icon} <b>Share</b> in Safari.', { icon: <ShareGlyph /> }); tr('Sign <r>in</r>') highlights "in" in red
// (pass r: (text) => ... to draw the highlight differently).
type Part = ReactNode | ((text: string) => ReactNode)
export function tr(en: string, parts: Record<string, Part> = {}, vars?: Record<string, string | number>): ReactNode {
  const s = t(en, vars)
  const out: ReactNode[] = []
  const re = /<b>(.*?)<\/b>|<r>(.*?)<\/r>|\{(\w+)\}/g
  let last = 0
  let m: RegExpExecArray | null
  let k = 0
  while ((m = re.exec(s))) {
    if (m.index > last) out.push(s.slice(last, m.index))
    if (m[1] !== undefined) out.push(<b key={k++}>{m[1]}</b>)
    else if (m[2] !== undefined) {
      const r = parts.r
      out.push(<Fragment key={k++}>{typeof r === 'function' ? r(m[2]) : <span className="text-red-ink">{m[2]}</span>}</Fragment>)
    } else {
      const p = parts[m[3]]
      out.push(<Fragment key={k++}>{typeof p === 'function' ? m[0] : p ?? m[0]}</Fragment>)
    }
    last = re.lastIndex
  }
  if (last < s.length) out.push(s.slice(last))
  return <>{out}</>
}
