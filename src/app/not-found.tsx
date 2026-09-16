import Link from 'next/link';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';

export default function NotFound() {
  return <><Navbar/><main style={{maxWidth:760,margin:'0 auto',padding:'150px 24px 100px',color:'var(--ink)'}}>
    <p style={{letterSpacing:'.15em'}}>404 / PAGE NOT AVAILABLE</p>
    <h1 style={{fontSize:44,margin:'24px 0'}}>This page is no longer here.</h1>
    <p>このページは削除されたか、見つかりません。Home・Vera・Appsからお探しください。</p>
    <nav aria-label="Available pages" style={{display:'flex',flexWrap:'wrap',gap:24,marginTop:32}}>
      <Link href="/">Home / Cleanroom</Link><Link href="/vera/">Vera</Link><Link href="/apps/">Apps</Link>
      <a href="https://github.com/Ag3497120/cleanroom">GitHub / Cleanroom</a>
    </nav>
  </main><Footer/></>;
}
