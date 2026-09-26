import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { site } from '../../content/site';

/** True once the profile photo has been added to public/. Checked at build time. */
export const hasPhoto = existsSync(join(process.cwd(), 'public', site.photo));
