// Load the language first, then the app: every screen's text is ready before anything renders.
import { initLang } from './i18n'

void initLang().then(() => import('./boot'))
