'use client';
import {useEffect,useRef,useState} from 'react';
import AskAI from './cleanroom-ask';
type Item={id:number;kind:string;text:string;source:string};
export default function CleanroomPlayground({locale}:{locale:string}){
  const ja=locale==='ja';const [mode,setMode]=useState<'agent'|'memo'|'search'>('agent');
  const [draft,setDraft]=useState(''),[memo,setMemo]=useState(''),[query,setQuery]=useState('');
  const [items,setItems]=useState<Item[]>([{id:1,kind:'DEMO / OWNER INTENT',text:'Offline first',source:'Scripted example, not your decision'}]);
  const [messages,setMessages]=useState<string[]>([ja?'操作デモです。依頼はこのタブだけに残ります。実AIへ渡す場合は「AIに頼む」を開きます。':'Interaction preview. Requests stay in this tab. Ask your AI explicitly calls your endpoint or prepares a handoff.']);
  const [selection,setSelection]=useState(0),[suppressed,setSuppressed]=useState(false);
  const [transfer,setTransfer]=useState(0),[menu,setMenu]=useState(false),[ask,setAsk]=useState(false);
  const [selected,setSelected]=useState<string[]>([]);
  const root=useRef<HTMLDivElement>(null),agentInput=useRef<HTMLTextAreaElement>(null),ownerInput=useRef<HTMLTextAreaElement>(null);
  const agentBody=useRef<HTMLDivElement>(null),ownerBody=useRef<HTMLDivElement>(null);
  const modeRef=useRef(mode);modeRef.current=mode;
  const timers=useRef<ReturnType<typeof setTimeout>[]>([]);
  useEffect(()=>()=>timers.current.forEach(clearTimeout),[]);
  useEffect(()=>{
    const node=root.current;if(!node)return;
    const wheel=(e:WheelEvent)=>{const target=modeRef.current==='agent'?agentBody.current:ownerBody.current;if(!target)return;e.preventDefault();target.scrollBy({top:e.deltaY*(e.deltaMode===1?18:e.deltaMode===2?target.clientHeight:1)});};
    node.addEventListener('wheel',wheel,{passive:false});return()=>node.removeEventListener('wheel',wheel);
  },[]);
  const activate=(next:typeof mode)=>{setMode(next);if(next==='agent')agentInput.current?.focus();else ownerInput.current?.focus();};
  const cycle=()=>activate(mode==='agent'?'memo':mode==='memo'?'search':'agent');
  const word=draft.match(/[^\s〈〉]{2,}$/)?.[0]??'';
  const matches=mode==='agent'&&!suppressed&&word?items.filter(x=>x.text.toLocaleLowerCase().startsWith(word.toLocaleLowerCase())).slice(0,8):[];
  const pick=(item:Item)=>{setDraft(draft.slice(0,-word.length)+'〈'+item.text+'〉 ');setSelected(s=>s.includes(item.text)?s:[...s,item.text]);setSuppressed(true);setTransfer(v=>v+1);agentInput.current?.focus();};
  const append=(kind:string,text:string,source:string)=>setItems(x=>[...x,{id:Date.now()+Math.random(),kind,text,source}]);
  const send=()=>{
    if(mode==='search'){if(!query.trim())cycle();return;}
    if(mode==='memo'){if(!memo.trim()){cycle();return;}append('YOUR MEMO',memo,'This tab only · not sent');setMemo('');return;}
    if(matches.length){pick(matches[Math.min(selection,matches.length-1)]);return;}
    if(!draft.trim()){cycle();return;}
    append('YOUR REQUEST',draft,'Your input · not executed');
    setMessages(x=>[...x,'> '+draft,ja?'依頼は未実行です。「AIに頼む」で送信先を選ぶか、ローカルCLIで進めてください。':'Request recorded, not executed. Choose Ask your AI or continue in the local CLI.']);
    setDraft('');setTransfer(v=>v+1);
  };
  const replay=()=>{
    timers.current.forEach(clearTimeout);
    setItems([{id:1,kind:'DEMO / OWNER INTENT',text:'Offline first',source:'Scripted fixture'}]);
    setMessages([ja?'記録デモを再生します。モデルもツールも呼びません。':'Replaying fixtures. No models or tools are called.']);
    const stages=[['AGENT','Inspecting an offline notes example.'],['AI ASSUMPTION','One device for now; multi-device sync is not assumed.'],['UNKNOWN','Actual multi-device conflict behavior has not been checked.'],['REVIEW / PROPOSAL','Offline storage: keep a recoverable copy before adding sync.'],['DELEGATE / PROPOSAL','Routine note-card layout can remain delegated.']];
    stages.forEach(([kind,text],i)=>timers.current.push(setTimeout(()=>{if(kind==='AGENT')setMessages(x=>[...x,text]);else append('DEMO / '+kind,text,'Scripted fixture · not evidence or mastery');setTransfer(v=>v+1);},700+i*800)));
  };
  const onKey=(e:React.KeyboardEvent<HTMLTextAreaElement>)=>{
    if(e.nativeEvent.isComposing)return;
    if(e.key==='F2'){e.preventDefault();setMenu(v=>!v);return;}
    if(e.key==='F3'){e.preventDefault();(mode==='agent'?agentBody:ownerBody).current?.focus();return;}
    if(e.key==='PageDown'||e.key==='PageUp'){e.preventDefault();const b=(mode==='agent'?agentBody:ownerBody).current;b?.scrollBy({top:(e.key==='PageDown'?1:-1)*b.clientHeight*.8});return;}
    if(e.key==='Escape'){setSuppressed(true);setMenu(false);return;}
    if(mode==='agent'&&matches.length&&(e.key==='ArrowDown'||e.key==='ArrowUp')){e.preventDefault();setSelection(s=>(s+(e.key==='ArrowDown'?1:-1)+matches.length)%matches.length);return;}
    if(mode==='agent'&&matches.length&&e.key==='Tab'){e.preventDefault();pick(matches[Math.min(selection,matches.length-1)]);return;}
    if(e.key==='Enter'&&!e.shiftKey&&!e.ctrlKey&&!e.metaKey){e.preventDefault();send();}
  };
  const filtered=query&&mode==='search'?items.filter(x=>(x.text+' '+x.kind).toLocaleLowerCase().includes(query.toLocaleLowerCase())):items;
  return <section className="cr-lab" aria-label="Agent and Owner interactive preview">
    <header><span>INTERACTIVE NOTEBOOK</span><span className="cr-state">{ja?'タブ内のみ / 未実行':'IN THIS TAB / NO EXECUTION'}</span><button onClick={replay}>{ja?'記録デモを再生':'Play the example'}</button><button onClick={()=>setAsk(true)}>{ja?'AIに頼む':'Ask your AI'} <small>BETA</small></button></header>
    <div ref={root} className={'cr-split active-'+mode}>
      <section className="cr-agent"><h3>Agent <small>{ja?'作業と回答':'Work & responses'}</small></h3><div ref={agentBody} className="cr-scroll" tabIndex={0} aria-label="Agent records">{messages.map((m,i)=><p key={i}>{m}</p>)}</div>
        <div className="cr-input"><label htmlFor="cr-agent-input">{ja?'一文で仕事を頼む':'What shall we build?'}</label><textarea id="cr-agent-input" ref={agentInput} value={draft} rows={3} maxLength={12000} onFocus={()=>setMode('agent')} onChange={e=>{setDraft(e.target.value);setSuppressed(false);setSelection(0);}} onKeyDown={onKey} placeholder={ja?'依頼、またはOwner項目の先頭2文字…':'A request, or the first letters of an Owner item…'}/>
          {matches.length>0&&<div className="cr-completions" role="listbox" aria-label="Owner references">{matches.map((x,i)=><button key={x.id} role="option" aria-selected={i===selection} onMouseDown={e=>e.preventDefault()} onClick={()=>pick(x)}>{x.text}</button>)}</div>}
          <div><button onClick={send}>{ja?'入力を送る':'Submit input'}</button><small>Enter · Tab · F2</small></div></div>
      </section>
      <div className="cr-divider" aria-hidden="true"><span key={transfer} className={transfer?'cr-transfer':''}>{transfer?'→':'·'}</span></div>
      <section className="cr-owner"><h3>Owner <small>{ja?'自分の判断と記録':'Your decisions & notes'}</small></h3><div className={'cr-owner-input '+(mode==='search'?'is-search':'is-memo')}><label htmlFor="cr-owner-input">{mode==='search'?'SEARCH / LOCAL':'MEMO / LOCAL'}</label><textarea id="cr-owner-input" ref={ownerInput} rows={2} maxLength={4000} value={mode==='search'?query:memo} onFocus={()=>{if(mode==='agent')setMode('memo');}} onChange={e=>mode==='search'?setQuery(e.target.value):setMemo(e.target.value)} onKeyDown={onKey} placeholder={mode==='search'?(ja?'このノート内を検索':'Search this notebook'):(ja?'自分用に残しておく':'A note just for you')}/><button onClick={cycle}>{ja?'入力先を切替':'Switch input'} ↵</button></div>
        <div ref={ownerBody} className="cr-scroll" tabIndex={0} aria-label="Owner records">{filtered.map(x=><article key={x.id}><small>{x.kind}</small><p>{x.text}</p><span>{x.source}</span></article>)}{!filtered.length&&<p>{ja?'一致する記録はありません。能力についての判断ではありません。':'No matching records. This is not an ability judgment.'}</p>}</div>
      </section>
    </div>
    <footer><span>{ja?'空欄Enter: Agent → メモ → 検索':'Empty Enter: Agent → Memo → Search'}</span><span>{ja?'スクロール: 入力先の側だけ':'Scroll: active input side only'}</span></footer>
    {menu&&<div className="cr-inline-menu"><button onClick={()=>{cycle();setMenu(false);}}>{ja?'入力先を切り替える':'Switch input'}</button><button onClick={()=>{setAsk(true);setMenu(false);}}>{ja?'AIに頼む':'Ask your AI'}</button><a href="#install">{ja?'CLIを導入':'Install the real CLI'}</a><button onClick={()=>setMenu(false)}>Esc / Close</button></div>}
    {ask&&<AskAI locale={locale} request={draft||[...items].reverse().find(x=>x.kind==='YOUR REQUEST')?.text||''} references={selected} onClose={()=>setAsk(false)} onResult={(answer,proposals)=>{setMessages(x=>[...x,'AI / UNVERIFIED ANSWER\n'+answer]);proposals.forEach(x=>append(x.kind+' / AI PROPOSAL',x.text,x.source));setTransfer(v=>v+1);}}/>}
  </section>;
}
