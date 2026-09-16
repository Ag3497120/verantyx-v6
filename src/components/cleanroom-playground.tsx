'use client';
import {useEffect,useRef,useState,type CSSProperties,type KeyboardEvent} from 'react';
import AskAI from './cleanroom-ask';
import {copy,en} from './cleanroom-playground-copy';

type Mode='agent'|'memo'|'search';
type Item={id:string;kind:'request'|'memo'|'learning'|'assumption'|'unknown';text:string;detail:string;source:string;at:string};
type Message={id:number;role:'user'|'assistant'|'system';text:string;fixture?:boolean};
type Pending={kind:'diff'|'instruction';text:string;choice:number};
const COMMANDS=['/help','/model','/insights','/details','/verantyx new','/verantyx cleanroom','/compact','/tutorial','/close'];
const DIFF='--- a/notes/storage.ts\n+++ b/notes/storage.ts\n@@ local copy before sync @@\n- await sync(note)\n+ await saveLocalCopy(note)\n+ await sync(note)';

export default function CleanroomPlayground({locale}:{locale:string}){
  const l=copy[locale]??en;
  const [mode,setMode]=useState<Mode>('agent'),[draft,setDraft]=useState(''),[memo,setMemo]=useState(''),[query,setQuery]=useState('');
  const [items,setItems]=useState<Item[]>([]),[messages,setMessages]=useState<Message[]>([]);
  const [selection,setSelection]=useState(0),[suppressed,setSuppressed]=useState(false),[transfer,setTransfer]=useState(0);
  const [ask,setAsk]=useState(false),[settings,setSettings]=useState(false),[details,setDetails]=useState(false);
  const [selected,setSelected]=useState<string[]>([]),[busy,setBusy]=useState(false),[pending,setPending]=useState<Pending|null>(null);
  const [guide,setGuide]=useState(true),[step,setStep]=useState(0),[reading,setReading]=useState(17),[conversation,setConversation]=useState(1);
  const [temporary,setTemporary]=useState<{question:string;answer:string}|null>(null),[omitted,setOmitted]=useState(false);
  const root=useRef<HTMLDivElement>(null),agentInput=useRef<HTMLTextAreaElement>(null),ownerInput=useRef<HTMLTextAreaElement>(null);
  const agentBody=useRef<HTMLDivElement>(null),ownerBody=useRef<HTMLDivElement>(null);
  const sequence=useRef(1),timers=useRef<ReturnType<typeof setTimeout>[]>([]);
  const history=useRef<Record<Mode,string[]>>({agent:[],memo:[],search:[]});
  const recall=useRef<{mode:Mode;index:number;draft:string}|null>(null);
  const pendingRef=useRef<Pending|null>(null),reviewWaiting=useRef(false),followAgent=useRef(true);
  useEffect(()=>{pendingRef.current=pending;},[pending]);
  useEffect(()=>()=>timers.current.forEach(clearTimeout),[]);
  useEffect(()=>{
    if(followAgent.current)agentBody.current?.scrollTo({top:agentBody.current.scrollHeight});
  },[messages]);
  useEffect(()=>{
    const node=root.current;if(!node)return;
    const wheel=(event:WheelEvent)=>{
      if(event.ctrlKey||event.metaKey)return;
      const target=mode==='agent'?agentBody.current:ownerBody.current;if(!target)return;
      event.preventDefault();
      target.scrollBy({top:event.deltaY*(event.deltaMode===1?20:event.deltaMode===2?target.clientHeight:1)});
    };
    node.addEventListener('wheel',wheel,{passive:false});
    return()=>node.removeEventListener('wheel',wheel);
  },[mode]);
  const later=(delay:number,action:()=>void)=>{timers.current.push(setTimeout(action,delay));};
  const activate=(next:Mode)=>{
    setMode(next);
    requestAnimationFrame(()=>{(next==='agent'?agentInput:ownerInput).current?.focus();});
  };
  const advance=(expected:number)=>setStep(current=>current===expected?Math.min(4,current+1):current);
  const cycle=()=>activate(mode==='agent'?'memo':mode==='memo'?'search':'agent');
  const message=(role:Message['role'],text:string,fixture=false)=>{
    const body=agentBody.current;
    followAgent.current=!body||body.scrollHeight-body.scrollTop-body.clientHeight<80;
    const row={id:sequence.current++,role,text,fixture};
    setMessages(previous=>{if(previous.length>=200)setOmitted(true);return [...previous,row].slice(-200);});
  };
  const append=(kind:Item['kind'],text:string,detail:string,source:string,id?:string)=>{
    const row={id:id??'N-'+String(sequence.current++).padStart(6,'0'),kind,text,detail,source,at:new Date().toISOString()};
    setItems(previous=>[...previous,row]);setTransfer(value=>value+1);
  };
  const remember=(key:Mode,value:string)=>{
    const entries=history.current[key];
    if(value.trim()&&entries.at(-1)!==value)entries.push(value);
    if(entries.length>100)entries.shift();
    recall.current=null;
  };
  const word=draft.match(/[^\s〈〉]{2,}$/)?.[0]??'';
  const references=mode==='agent'&&!suppressed&&!pending&&!draft.trimStart().startsWith('/')&&word
    ?items.filter(item=>item.text.toLocaleLowerCase().startsWith(word.toLocaleLowerCase())).slice(0,8):[];
  const commandMatches=mode==='agent'&&!pending&&!suppressed&&draft.startsWith('/')&&!draft.startsWith('//')
    ?COMMANDS.filter(command=>command.startsWith(draft)).slice(0,8):[];
  const completions=[...commandMatches.map(command=>({key:command,text:command,item:null as Item|null})),
    ...references.map(item=>({key:item.id,text:item.text,item}))];
  const pick=(entry:typeof completions[number])=>{
    if(entry.item){
      setDraft(draft.slice(0,-word.length)+'〈'+entry.text+'〉 ');
      setSelected(previous=>previous.includes(entry.text)?previous:[...previous,entry.text]);
      setTransfer(value=>value+1);advance(3);
    }else setDraft(entry.text);
    setSuppressed(true);agentInput.current?.focus();
  };
  const finishExample=(allowed:boolean)=>{
    setBusy(false);setPending(null);reviewWaiting.current=false;
    message('assistant',allowed?l.finished:l.refused,true);
    if(allowed)append('unknown',l.uncertainty,'',l.source);
  };
  const choose=(index:number)=>{
    if(!pending)return;
    const current=pending;setPending(null);
    if(current.kind==='diff'){finishExample(index!==3);return;}
    message('user',current.text);
    message('system',index===0?l.queued:l.steered,true);
    if(reviewWaiting.current){reviewWaiting.current=false;later(150,()=>setPending({kind:'diff',text:DIFF,choice:0}));}
  };
  const replay=()=>{
    if(busy)return;
    setTemporary(null);setSettings(false);setBusy(true);advance(0);
    message('user',l.demoRequest,true);
    append('request',l.demoRequest,'',l.source);
    const identity='L-'+String(sequence.current++).padStart(6,'0');
    later(900,()=>message('assistant',l.demoPlan,true));
    later(1800,()=>append('learning',l.topic,l.insight,l.source,identity));
    later(3400,()=>{
      if(pendingRef.current){reviewWaiting.current=true;return;}
      setPending({kind:'diff',text:DIFF,choice:0});
    });
  };
  const command=(value:string)=>{
    if(value==='verantyx'){setGuide(false);message('system',l.guideDone);return true;}
    if(!value.startsWith('/')||value.startsWith('//'))return false;
    const normalized=value.replace(/^\/(?:verantyx|vernatyx)(?:\s+|$)/,'/').trim();
    if(['/tutorial'].includes(normalized)){setGuide(true);setStep(0);return true;}
    if(['/close','/done'].includes(normalized)){setSettings(false);setTemporary(null);if(normalized==='/done')setGuide(false);return true;}
    if(['/','/setup','/model','/help','/commands'].includes(normalized)){setSettings(true);return true;}
    if(normalized==='/details'){setDetails(previous=>!previous);return true;}
    if(normalized==='/insights'){activate('memo');return true;}
    if(normalized==='/new'){
      if(busy){message('system',l.pending);return true;}
      setMessages([]);setConversation(value=>value+1);setSelected([]);setTemporary(null);
      history.current.agent=[];recall.current=null;setOmitted(false);
      message('system',l.newAgent);return true;
    }
    if(normalized.startsWith('/cleanroom')||normalized==='/compact'){message('system',l.localOnly);return true;}
    message('system',l.commands+': '+COMMANDS.join('  '));return true;
  };
  const send=()=>{
    if(mode==='search'){
      if(!query.trim())cycle();else{remember('search',query);advance(2);}
      return;
    }
    if(mode==='memo'){
      if(!memo.trim()){cycle();return;}
      if(memo.startsWith('/verantyx ')||memo.startsWith('/vernatyx ')){
        // Owner commands here are explanations only; never silently replace the notebook.
        message('system',l.localOnly);setMemo('');return;
      }
      append('memo',memo,'',l.local);remember('memo',memo);setMemo('');advance(1);return;
    }
    if(pending){choose(pending.choice);return;}
    if(completions.length&&!suppressed){pick(completions[Math.min(selection,completions.length-1)]);return;}
    const value=draft.trim();
    if(!value){cycle();return;}
    if(command(value)){setDraft('');setSuppressed(false);return;}
    if(value.startsWith('//')){
      setSettings(false);setTemporary({question:value.slice(2).trim(),answer:l.privateAnswer});
      setDraft('');advance(4);return;
    }
    const identity=value.match(/^(L-\d+)(?:\s+([\s\S]*))?$/i);
    const insight=identity?items.find(item=>item.id===identity[1].toUpperCase()&&item.kind==='learning'):undefined;
    if(insight){
      message('user',value);message('assistant',l.question+'\n\n'+insight.detail,true);
      remember('agent',value);setDraft('');return;
    }
    setSettings(false);setTemporary(null);
    if(busy){setPending({kind:'instruction',text:value,choice:0});setDraft('');remember('agent',value);return;}
    message('user',value);message('system',l.notRun);
    append('request',value,'',l.local);remember('agent',value);
    setDraft('');setTransfer(value=>value+1);advance(0);
  };
  const onKey=(event:KeyboardEvent<HTMLTextAreaElement>)=>{
    if(event.nativeEvent.isComposing)return;
    if(event.key==='F2'){event.preventDefault();setSettings(value=>!value);return;}
    if(event.key==='F3'){event.preventDefault();(mode==='agent'?agentBody:ownerBody).current?.focus();return;}
    if(event.key==='PageDown'||event.key==='PageUp'){
      event.preventDefault();const body=(mode==='agent'?agentBody:ownerBody).current;
      body?.scrollBy({top:(event.key==='PageDown'?1:-1)*body.clientHeight*.8});return;
    }
    if(event.key==='Escape'){setSuppressed(true);setSettings(false);return;}
    if(mode==='agent'&&pending&&(event.key==='ArrowDown'||event.key==='ArrowUp')){
      event.preventDefault();const count=pending.kind==='diff'?4:2;
      setPending({...pending,choice:(pending.choice+(event.key==='ArrowDown'?1:-1)+count)%count});return;
    }
    if(mode==='agent'&&completions.length&&(event.key==='ArrowDown'||event.key==='ArrowUp')){
      event.preventDefault();setSelection(index=>(index+(event.key==='ArrowDown'?1:-1)+completions.length)%completions.length);return;
    }
    if(mode==='agent'&&completions.length&&event.key==='Tab'){
      event.preventDefault();pick(completions[Math.min(selection,completions.length-1)]);return;
    }
    if((event.key==='ArrowUp'||event.key==='ArrowDown')&&!pending){
      const element=event.currentTarget,value=element.value,position=element.selectionStart;
      const edge=event.key==='ArrowUp'?!value.slice(0,position).includes('\n'):!value.slice(position).includes('\n');
      const entries=history.current[mode];
      if(edge&&entries.length){
        const current=recall.current?.mode===mode?recall.current:{mode,index:entries.length,draft:value};
        const index=Math.max(0,Math.min(entries.length,current.index+(event.key==='ArrowUp'?-1:1)));
        if(index!==current.index){
          event.preventDefault();recall.current={...current,index};
          const recalled=index===entries.length?current.draft:entries[index];
          if(mode==='agent')setDraft(recalled);else if(mode==='memo')setMemo(recalled);else setQuery(recalled);
          setSuppressed(true);requestAnimationFrame(()=>element.setSelectionRange(recalled.length,recalled.length));return;
        }
      }
    }
    if(event.key==='Enter'&&!event.shiftKey&&!event.ctrlKey&&!event.metaKey&&!event.altKey){event.preventDefault();send();}
  };
  const filtered=mode==='search'&&query?items.filter(item=>(item.text+' '+item.detail).toLocaleLowerCase().includes(query.toLocaleLowerCase())):items;
  const kinds={request:l.intent,memo:l.note,learning:l.proposal,assumption:l.assumption,unknown:l.unknown};
  const hints=[l.hint0,l.hint1,l.hint2,l.hint3,l.hint4];
  const currentRequest=draft||[...items].reverse().find(item=>item.kind==='request')?.text||'';
  return <section className="cr-lab" aria-label={l.preview} style={{'--cr-reader-size':reading+'px'} as CSSProperties}>
    <header><strong>{l.preview}</strong><span className="cr-state">{l.demoModel}</span>
      <button onClick={replay} disabled={busy}>{l.play}</button><button onClick={()=>setAsk(true)}>{l.ask} <small>BETA</small></button></header>
    <p className="cr-rehearsal-boundary">{l.boundary}</p>
    <nav className="cr-pane-tools" aria-label={l.switch}>
      <button aria-pressed={mode==='agent'} onClick={()=>activate('agent')}>Agent</button>
      <button aria-pressed={mode==='memo'} onClick={()=>activate('memo')}>Owner · {l.note}</button>
      <button aria-pressed={mode==='search'} onClick={()=>activate('search')}>Owner · {l.search}</button>
      <span className="cr-text-tools"><span>{l.size}</span><button aria-label={l.smaller} onClick={()=>setReading(value=>Math.max(16,value-1))} disabled={reading<=16}>A−</button><button aria-label={l.larger} onClick={()=>setReading(value=>Math.min(22,value+1))} disabled={reading>=22}>A+</button></span>
    </nav>
    {guide&&<aside className="cr-guide" aria-live="polite"><strong>{l.guide} · {step+1}/5</strong><p>{hints[step]}</p><button onClick={()=>setGuide(false)}>{l.skip}</button></aside>}
    <div ref={root} className={'cr-split active-'+mode}>
      <section className="cr-agent" aria-label="Agent"><h3>Agent <small>{l.agent} · #{conversation}</small></h3>
        <div ref={agentBody} className="cr-scroll" tabIndex={-1} aria-label={l.agent}>
          {omitted&&<p className="cr-system-note">{l.historyLimit}</p>}
          {!messages.length&&<p className="cr-system-note">{l.request}</p>}
          {messages.filter(row=>row.role!=='system').map(row=><article className={'cr-message cr-message-'+row.role} key={row.id}>
            <small>{row.role==='user'?l.you:l.ai}{row.fixture?' · '+l.example:''}</small><p>{row.text}</p>
          </article>)}
          {temporary&&<aside className="cr-temporary"><strong>{l.private}</strong><p>{temporary.question}</p><p>{temporary.answer}</p><small>{l.privateHelp}</small><button onClick={()=>setTemporary(null)}>{l.close}</button></aside>}
        </div>
        <details className="cr-system" open={details} onToggle={event=>setDetails(event.currentTarget.open)}>
          <summary>{l.system} · {messages.filter(row=>row.role==='system').length}</summary>
          {messages.filter(row=>row.role==='system').map(row=><p key={row.id}>{row.text}</p>)}
        </details>
        {settings&&<aside className="cr-settings"><strong>{l.settings}</strong><p>{l.settingsHelp}</p><button onClick={()=>setAsk(true)}>{l.model}</button><button onClick={()=>setSettings(false)}>{l.close}</button><div className="cr-command-list">{COMMANDS.map(value=><button key={value} onClick={()=>{setDraft(value);setSuppressed(true);activate('agent');}}><code>{value}</code></button>)}</div></aside>}
        <div className={'cr-input'+(busy?' is-working':'')}>
          {busy&&<output className="cr-progress"><span className="cr-spinner" aria-hidden="true"/> {l.eta}</output>}
          {pending&&<fieldset className="cr-inline-choice"><legend>{pending.kind==='diff'?l.diff:l.scheduling}</legend>
            {pending.kind==='diff'&&<pre className="cr-diff">{pending.text.split('\n').map((line,index)=><span key={index} className={line.startsWith('+')?'cr-diff-add':line.startsWith('-')?'cr-diff-remove':''}>{line+'\n'}</span>)}</pre>}
            <div role="radiogroup" aria-label={pending.kind==='diff'?l.diff:l.scheduling}>
              {(pending.kind==='diff'?[l.once,l.workspace,l.always,l.deny]:[l.queue,l.steer]).map((label,index)=><label key={label} className={pending.choice===index?'is-selected':''}>
                <input type="radio" name="cr-review-choice" checked={pending.choice===index}
                  onChange={()=>setPending({...pending,choice:index})}
                  onKeyDown={event=>{if(event.key==='Enter'){event.preventDefault();choose(index);}}}/>
                <span>{label}</span></label>)}
            </div><small>{l.diffDetail}</small>
          </fieldset>}
          <label htmlFor="cr-agent-input">{l.request}</label>
          <textarea id="cr-agent-input" ref={agentInput} rows={3} maxLength={12000} value={draft} readOnly={!!pending}
            onFocus={()=>setMode('agent')} onChange={event=>{setDraft(event.target.value);setSuppressed(false);setSelection(0);recall.current=null;}}
            onKeyDown={onKey} placeholder={l.placeholder}/>
          {completions.length>0&&<ul className="cr-completions" aria-label={l.refs}>{completions.map((entry,index)=><li key={entry.key}><button data-selected={index===selection} onMouseDown={event=>event.preventDefault()} onClick={()=>pick(entry)}>{entry.text}</button></li>)}</ul>}
          <div><button onClick={()=>{if(pending)choose(pending.choice);else{activate('agent');if(mode==='agent')send();}}}>{pending?l.submit:l.submit}</button><small>Enter · Tab · /help</small></div>
          {guide&&<p className="cr-tour-start">{l.start}</p>}
        </div>
      </section>
      <div className="cr-divider" aria-hidden="true"><span key={transfer} className={transfer?'cr-transfer':''}>{transfer?'→':'·'}</span></div>
      <section className="cr-owner" aria-label="Owner"><h3>Owner <small>{l.owner}</small></h3>
        <div className={'cr-owner-input '+(mode==='search'?'is-search':'is-memo')}>
          <label htmlFor="cr-owner-input">{mode==='search'?l.search:l.note}</label>
          <textarea id="cr-owner-input" ref={ownerInput} rows={2} maxLength={4000} value={mode==='search'?query:memo}
            onFocus={()=>{if(mode==='agent')setMode('memo');}}
            onChange={event=>{if(mode==='search'){setQuery(event.target.value);advance(2);}else setMemo(event.target.value);recall.current=null;}}
            onKeyDown={onKey} placeholder={mode==='search'?l.search:l.memo}/>
          <button onClick={cycle}>{l.switch} ↵</button>
        </div>
        {pending&&<output className="cr-owner-alert">{l.pending}</output>}
        <div ref={ownerBody} className="cr-scroll" tabIndex={-1} aria-label={l.owner}>
          {filtered.map(item=><article key={item.id} className={'cr-note cr-note-'+item.kind}>
            <small>{kinds[item.kind]}</small><h4>{item.text}</h4>
            {item.kind==='learning'&&<><button className="cr-insight-id" onClick={()=>{setDraft(item.id);setSuppressed(true);activate('agent');}}>{item.id}</button><p>{l.next}</p></>}
            {item.kind!=='learning'&&item.detail&&<p>{item.detail}</p>}
            <span>{item.source}</span><time dateTime={item.at}>{l.time} · {new Date(item.at).toLocaleString(locale)}</time>
          </article>)}
          {!filtered.length&&<p className="cr-system-note">{l.empty}</p>}
        </div>
      </section>
    </div>
    <footer><span>{l.controls}</span><span>{l.scroll}</span><a href="#install">{l.installed}</a></footer>
    {ask&&<AskAI locale={locale} request={temporary?.question??currentRequest} references={temporary?[]:selected} onClose={()=>setAsk(false)}
      onResult={(answer,proposals)=>{
        if(temporary){setTemporary({...temporary,answer});return;}
        message('assistant',answer);message('system','AI / unverified answer');
        proposals.forEach(proposal=>append('learning',proposal.text,proposal.kind+' / '+proposal.text,proposal.source,'L-'+String(sequence.current++).padStart(6,'0')));
      }}/>}
  </section>;
}
