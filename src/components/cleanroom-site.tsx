'use client';
import {useEffect,useState,useSyncExternalStore,type ReactNode} from 'react';
import Image from 'next/image';
import Link from 'next/link';
import copy from './cleanroom-copy.json';
import CleanroomPlayground from './cleanroom-playground';
const REPO='https://github.com/Ag3497120/cleanroom';
function readLocale(){
  const value=new URLSearchParams(location.search).get('lang');
  return value&&copy.some(x=>x.code===value)?value:'en';
}
function subscribeLocale(notify:()=>void){
  window.addEventListener('popstate',notify);
  return()=>window.removeEventListener('popstate',notify);
}
function setLocale(value:string){
  const url=new URL(location.href);url.searchParams.set('lang',value);
  history.replaceState(history.state,'',url);window.dispatchEvent(new PopStateEvent('popstate'));
}
export default function CleanroomSite({base='',marketing=false,author=false,themeControl}:{base?:string;marketing?:boolean;author?:boolean;themeControl?:ReactNode}){
  const locale=useSyncExternalStore(subscribeLocale,readLocale,()=>'en');
  const [platform,setPlatform]=useState('macOS'),[copied,setCopied]=useState(false),[copyError,setCopyError]=useState('');
  const l=copy.find(x=>x.code===locale)??copy[0],ja=locale==='ja';
  useEffect(()=>{document.documentElement.lang=locale;},[locale]);
  const command='git clone '+REPO+'.git cleanroom\ncd cleanroom\npython3 -m venv .venv\nsource .venv/bin/activate\npython -m pip install -e ./core\nverantyx setup\nverantyx';
  const copyInstall=async()=>{try{await navigator.clipboard.writeText(command);setCopied(true);setCopyError('');}catch{setCopyError(ja?'自動コピーできません。下のコマンドを選択できます。':'Copy unavailable. Select the commands below.');}};
  const guide=REPO+'/blob/main/core/docs/OPERATIONS.'+locale+'.md';
  const asset=(path:string)=>base+'/cleanroom/'+path;

  const revised:Record<string,{title:string;body:string;link:string;gif:string;static:string}> = {
    en:{title:'Understanding can begin while work continues.',body:'Keep small, optional explanations at safe work boundaries. Send an L-number to ask about one without replacing the task. AI time estimates are ranges, not deadlines; a missing estimate stays unknown.',link:'Live learning, sessions and permissions',gif:'Open the English GIF walkthrough',static:'Static image / reduced motion'},
    ja:{title:'完成を待たず、作りながら理解を残す。',body:'仕事の区切りで、必要な説明だけを任意のノートへ。L番号を送れば元の仕事を置き換えずに質問できます。AIの予想時間は幅のある見積もりで、未取得なら不明と表示します。',link:'作業中の学び・セッション・権限',gif:'英語のGIF操作デモを開く',static:'静止画 / 動きを減らす'},
    'zh-Hans':{title:'不必等到完成，边做边留下理解。',body:'在安全节点保存少量可选解释。发送L编号提问，不替换原任务。AI估时是范围而不是截止时间；没有估时就明确显示未知。',link:'工作中的学习、会话与权限',gif:'打开英语GIF操作演示',static:'静态图片 / 减少动态效果'},
    ko:{title:'완료를 기다리지 않고, 만드는 중에 이해를 남깁니다.',body:'안전한 작업 구간에서 작은 설명을 선택적으로 남깁니다. L번호로 원래 작업을 바꾸지 않고 질문합니다. AI 예상 시간은 범위이며, 없으면 모른다고 표시합니다.',link:'작업 중 학습, 세션과 권한',gif:'영어 GIF 조작 안내 열기',static:'정지 이미지 / 움직임 줄이기'},
    es:{title:'Comprender no tiene que esperar al final.',body:'Guarda explicaciones breves y opcionales en pasos seguros. Envía un número L para preguntar sin reemplazar la tarea. Las estimaciones de IA son rangos, no plazos; si no existen, se indica que se desconocen.',link:'Aprendizaje durante el trabajo, sesiones y permisos',gif:'Abrir el recorrido GIF en inglés',static:'Imagen estática / menos movimiento'},
  };
  const update=revised[locale]??revised.en;

  const install=<section className="cr-install" id="install"><div><span className="cr-index">02 / ON YOUR COMPUTER</span><h2>{l.start}</h2><p>Python 3.11+ · macOS / Linux · Windows via WSL2</p></div><div className="cr-terminal-card"><header><fieldset className="cr-platforms" aria-label="Installation platform">{['macOS','Linux','Windows / WSL2'].map(p=><button aria-pressed={platform===p} key={p} onClick={()=>{setPlatform(p);setCopied(false);}}>{p}</button>)}</fieldset><button className="cr-copy" onClick={()=>void copyInstall()} aria-label={ja?'インストールコマンドをコピー':'Copy install commands'} title={ja?'コマンドをコピー':'Copy commands'}><Image unoptimized src={asset('mark.svg')} width="30" height="30" alt=""/><span>{copied?(ja?'コピー済み':'Copied'):'Copy'}</span></button></header><pre><code>{command}</code></pre><p>{platform==='Windows / WSL2'?(ja?'先にWSL2内でPythonとGitを用意します。ネイティブWindows版の動作保証ではありません。':'Run inside WSL2 with Python and Git. This is not a native Windows compatibility claim.'):(ja?'Python 3.11以上とGitが必要です。新しいターミナルでは仮想環境を再び有効化します。':'Requires Python 3.11+ and Git. Reactivate the environment in a new terminal.')}</p><output>{copyError}</output></div><p><a href={guide}>{l.read} ↗</a> · <a href={REPO}>GitHub ↗</a></p></section>;
  return <div className="cr-site" lang={locale}>
    <a className="cr-skip" href="#main">Skip to content</a>
    <nav className="cr-top"><Link className="cr-small-brand" href={base+'/?lang='+locale}><Image unoptimized src={asset('mark.svg')} width="32" height="32" alt="Cleanroom"/>cleanroom</Link><div>{marketing&&<><Link href="/">Home</Link><Link href="/vera/">Vera</Link><Link href="/apps/">Apps</Link></>}{themeControl}<a href={guide}>{l.read}</a><a href={REPO}>GitHub</a><label className="cr-language"><span className="cr-sr">Language</span><select value={locale} onChange={e=>setLocale(e.target.value)}>{copy.map(x=><option key={x.code} value={x.code}>{x.name}</option>)}</select></label></div></nav>
    <main id="main">
      {author?<article className="cr-author"><span className="cr-index">THE PERSON BEHIND THE NOTEBOOK</span><h1>{l.author}</h1><h2>motonishi kodai</h2><p>{l.origin}</p><p>{l.intro}</p><p>{ja?'これは、個人が自分の開発と将来への不安から始めたプロジェクトです。商用サービスを売るための紹介ではありません。AIに任せながらも、作るものへの愛着や、作り手としての理解を残したいと思っています。':'This began as a personal response to building with AI and worrying about the future. It is not a commercial sales pitch. I want to keep the attachment to what I make, and the understanding that lets me continue shaping it.'}</p><aside>{l.foot}</aside><a href={REPO+'/blob/main/core/docs/origins/README.md'}>{ja?'制作のきっかけと議論を読む':'Read the origins and development notes'} ↗</a></article>:<>
      <header className="cr-hero"><span className="cr-index">WORKSPACE + LEARNING NOTEBOOK + YOUR SKILLS</span><Image unoptimized className="cr-wordmark" src={asset('logo.svg')} width="550" height="135" alt="cleanroom"/><h1>{l.tag}</h1><p>{l.intro}</p><div className="cr-hero-actions"><a className="cr-primary" href="#play">{l.try}</a><a href="#install">{l.setup}</a><a href="#work">{l.quick}</a></div><span className="cr-edition">SOURCE PREVIEW · MIT · YOUR PACE, YOUR CHOICE</span></header>
      <section className="cr-play-section" id="play"><div className="cr-section-title"><span className="cr-index">01 / TRY THE FEEL</span><h2>{l.ops}</h2><p>{ja?'通信なしですぐ試せる操作デモ。実AIへの送信は「AIに頼む」から選択する別の操作です。':'Try the interaction without an account or network call. Ask your AI is a separate, explicit action.'}</p></div><CleanroomPlayground locale={locale}/><p className="cr-caption">{l.limits}</p></section>
      {install}
      <section className="cr-pillars" id="notebook"><div className="cr-section-title"><span className="cr-index">03 / THE EXPERIENCE STAYS WITH YOU</span><h2>{l.features}</h2><p>{l.full}</p></div><div className="cr-pillar-grid">{l.pillars.map(([title,body],index)=><article key={title}><span className="cr-index">0{index+1}</span><h3>{title}</h3><p>{body}</p></article>)}</div><aside className="cr-example"><p>{l.example}</p><p>{l.distinction}</p><p>{l.pace}</p><a href={guide}>{l.docs} ↗</a></aside></section>
      <aside className="cr-live-feature"><h3>{update.title}</h3><p>{update.body}</p><a href={REPO+'/blob/main/docs/live-learning-and-sessions.md'}>{update.link} ↗</a></aside>
      <section className="cr-film" id="work"><div className="cr-film-media"><video controls playsInline preload="none" poster={asset('cli-demo-poster.png')} aria-label="English CLI walkthrough with labelled scripted demo data"><source src={asset('cli-demo.mp4')} type="video/mp4"/><track kind="captions" src={asset('cli-demo.'+locale+'.vtt')} srcLang={locale} label={l.name} default/><a href={asset('cli-demo.gif')}>View GIF</a></video><details className="cr-gif"><summary>{update.gif}</summary><Image unoptimized src={asset('cli-demo.gif')} width={1300} height={1014} alt="English CLI: continuous conversation, optional L-number explanation, private memo and inline candidate review"/><a href={asset('cli-demo-poster.png')}>{update.static}</a></details><small>{ja?'英語設定の実CLIを操作・録画。作業結果は明示したデモデータで、モデル呼び出しはありません。':'English CLI, real key interactions. Work output is labelled fixture data; no model calls.'}</small></div><div><span className="cr-index">04 / WHAT STAYS WITH YOU?</span><h2>{l.known}</h2><p>{l.meaning}</p><ol>{l.items.map((x,i)=><li key={x}><span>0{i+1}</span>{x}</li>)}</ol><a href={guide}>{l.read} ↗</a></div></section>
      <section className="cr-notes"><div><span className="cr-index">05 / TWO SIDES, ONE PROJECT</span><h2>{ja?'任せることと、手元に残すこと。':'A place to work. A place to keep.'}</h2></div><article><h3>Agent</h3><p>{l.request}</p><p>{l.ref}</p></article><article><h3>Owner</h3><p>{l.memo}</p><p>{l.privacy}</p></article></section>
      <section className="cr-origin"><span className="cr-index">06 / WHY I STARTED</span><h2>{ja?'作る経験まで、手放したくなかった。':'I did not want to lose the experience of making.'}</h2><p>{l.origin}</p><div><a href={REPO+'/blob/main/core/docs/origins/README.md'}>{ja?'原文と開発記':'Sources & development notes'} ↗</a>{marketing&&<Link href={'/author/?lang='+locale}>{l.author} ↗</Link>}</div></section>
      <section className="cr-boundary"><h3>{ja?'試せること。まだ約束しないこと。':'What you can try. What we do not promise.'}</h3><p>{l.limits}</p><p>{ja?'APIキーやサブスクトークンをサイトに保存しません。API接続は明示操作で指定先へ直接送信します。提供先の保持方針や料金は別です。実際の開発やサブスクの利用はCLIで行えます。':'This site does not persist API keys or subscription tokens. Direct API calls require explicit consent. Provider retention and billing are separate. Use the CLI for actual development and official subscription connections.'}</p><a href={REPO+'/blob/main/docs/PUBLICATION.md'}>{ja?'公開構成とプライバシー':'Publication & privacy'} ↗</a>{!marketing&&<a href={base+'/terminal/'}>{ja?'既存の計算ゲートウェイ（別サービス）':'Existing compute gateway (separate service)'} ↗</a>}</section>
      </>}
    </main>
    <footer className="cr-site-footer"><Link className="cr-small-brand" href={base+'/'}><Image unoptimized src={asset('mark.svg')} alt="" width="32" height="32"/>cleanroom</Link><p>{l.foot}</p><a href={REPO}>GitHub</a><a href={REPO+'/blob/main/LICENSE'}>MIT</a>{marketing&&<Link href={'/author/?lang='+locale}>{l.author}</Link>}{marketing&&<Link href="/apps/">Other projects</Link>}</footer>
  </div>;
}
