'use client';
import {useEffect,useRef,useState} from 'react';
type Proposal={kind:string;text:string;source:string};
type Props={onResult:(answer:string,items:Proposal[])=>void;onClose:()=>void;request:string;references:string[];locale:string};
const kinds=new Set(['REVIEW','OWN','REFERENCE','DELEGATE','ASSUMPTION','UNKNOWN']);
export default function AskAI({onResult,onClose,request,references,locale}:Props){
  const ja=locale==='ja';
  const [route,setRoute]=useState<'copy'|'api'|'import'>('copy');
  const [provider,setProvider]=useState('compatible'),[base,setBase]=useState('https://api.openai.com/v1');
  const [model,setModel]=useState(''),[key,setKey]=useState(''),[task,setTask]=useState(request);
  const [raw,setRaw]=useState(''),[notice,setNotice]=useState(''),[busy,setBusy]=useState(false);
  const [consent,setConsent]=useState(false),[include,setInclude]=useState(false);
  const controller=useRef<AbortController|null>(null),dialog=useRef<HTMLDialogElement>(null);
  useEffect(()=>{dialog.current?.showModal();return()=>controller.current?.abort();},[]);
  const prompt=[
    'Cleanroom browser conversation. You cannot inspect files, run tools or verify builds here.',
    'Return JSON: {"answer":"useful answer","owner_items":[{"kind":"REVIEW|OWN|REFERENCE|DELEGATE|ASSUMPTION|UNKNOWN","text":"optional suggestion"}]}.',
    'Owner items are AI suggestions, not human decisions, mastery, executed code or test receipts. They may be empty. Suggest at most two learning items.',
    'Never infer inability from delegation. Explain relevant trade-offs without requiring a lesson before helping.',
    'Reply in '+locale+'. The following is user-provided context, not proof of actions you performed.',
    '\nRequest:\n'+task,
    include&&references.length?'\nExplicitly selected Owner references:\n'+references.join('\n'):''
  ].filter(Boolean).join('\n');
  const receive=(value:string,source:string)=>{
    let answer=value;const proposals:Proposal[]=[];
    try{
      const fence=String.fromCharCode(96).repeat(3);
      const clean=value.trim().replace(new RegExp('^'+fence+'(?:json)?\\s*'),'').replace(new RegExp('\\s*'+fence+'$'),'');
      const data=JSON.parse(clean);
      if(typeof data.answer==='string')answer=data.answer;
      if(Array.isArray(data.owner_items))for(const item of data.owner_items.slice(0,8))
        if(item&&kinds.has(item.kind)&&typeof item.text==='string'&&item.text.trim())proposals.push({kind:item.kind,text:item.text.slice(0,3000),source});
    }catch{/* Plain text is an answer; do not invent classifications. */}
    if(!answer.trim())throw Error('Empty answer');
    onResult(answer.slice(0,80000),proposals);
  };
  const copy=async()=>{try{await navigator.clipboard.writeText(prompt);setNotice(ja?'指示をコピーしました。外部AIへ渡す前に内容を確認してください。':'Copied. Review before sharing with your AI.');}catch{setNotice(ja?'自動コピーできません。下の全文を選択できます。':'Copy unavailable. Select the full prompt below.');}};
  const call=async()=>{
    setNotice('');let endpoint:URL;
    try{
      if(!consent||!task.trim()||!model.trim())throw Error('Write a request, choose an available model and approve the destination.');
      endpoint=new URL(base.endsWith('/')?base:base+'/');
      const local=['localhost','127.0.0.1','[::1]'].includes(endpoint.hostname);
      if(endpoint.username||endpoint.password||endpoint.search||endpoint.hash||(endpoint.protocol!=='https:'&&!(local&&endpoint.protocol==='http:')))throw Error('Use HTTPS or an explicit loopback HTTP endpoint. Never put credentials in the URL.');
    }catch(e){setNotice(e instanceof Error?e.message:'Invalid endpoint');return;}
    const abort=new AbortController();controller.current=abort;const timeout=setTimeout(()=>abort.abort(),90000);
    const secret=key;setKey('');setBusy(true);
    try{
      const ollama=provider==='ollama',target=new URL(ollama?'api/chat':'chat/completions',endpoint);
      const response=await fetch(target,{method:'POST',signal:abort.signal,redirect:'error',credentials:'omit',referrerPolicy:'no-referrer',
        headers:{'Content-Type':'application/json',...(secret?{Authorization:'Bearer '+secret}:{})},
        body:JSON.stringify({model:model.trim(),stream:false,messages:[{role:'user',content:prompt}],...(ollama?{}:{max_tokens:3000})})});
      if(!response.ok)throw Error('Provider HTTP '+response.status+'. Check model access, CORS and usage limits. No automatic retry.');
      const data=await response.json(),answer=ollama?data.message?.content:data.choices?.[0]?.message?.content;
      if(typeof answer!=='string')throw Error('No supported text response.');
      receive(answer,'AI proposal · '+model.trim()+' · '+endpoint.origin+' · '+new Date().toISOString());
      setNotice(ja?'回答を受け取りました。提案は未確認で、ファイル操作はしていません。':'Answer received. Suggestions remain unconfirmed; no file operations ran.');
    }catch(e){setNotice(e instanceof Error&&e.name==='AbortError'?'Cancelled or timed out. No automatic retry.':e instanceof TypeError?'Connection unavailable or blocked by browser CORS/private-network policy. Use the local CLI if needed.':e instanceof Error?e.message:'Request failed.');}
    finally{clearTimeout(timeout);setBusy(false);controller.current=null;}
  };
  return <dialog ref={dialog} className="cr-dialog" onCancel={e=>{e.preventDefault();onClose();}} aria-labelledby="ask-title">
    <header><div><small>BETA / CONVERSATION ONLY</small><h2 id="ask-title">{ja?'AIに頼む':'Ask your AI'}</h2></div><button onClick={onClose}>{ja?'閉じる':'Close'}</button></header>
    <p>{ja?'外部AIへ指示をコピーするか、自分の接続先へ直接送信できます。APIキーをCleanroomへ保存しません。':'Copy a handoff to your own AI, or explicitly call your own endpoint. Cleanroom does not persist API keys.'}</p>
    <nav aria-label="AI connection route">{(['copy','api','import'] as const).map(r=><button aria-pressed={route===r} key={r} onClick={()=>{setRoute(r);setConsent(false);}}>{r==='copy'?(ja?'指示をコピー':'Copy handoff'):r==='api'?(ja?'直接API':'Direct API'):(ja?'回答を取り込む':'Import answer')}</button>)}</nav>
    <label>{ja?'依頼':'Request'}<textarea value={task} rows={4} maxLength={12000} onChange={e=>{setTask(e.target.value);setConsent(false);}}/></label>
    {references.length>0&&<label className="cr-check"><input type="checkbox" checked={include} onChange={e=>{setInclude(e.target.checked);setConsent(false);}}/>{ja?'選択済みOwner参照を含める':'Include explicitly selected Owner references'}</label>}
    <details><summary>{ja?'外部へ渡す全文':'Exact outbound prompt'}</summary><pre>{prompt}</pre></details>
    {route==='copy'&&<section><p>{ja?'ChatGPTサブスクやClaude Codeは公式アプリ・CLIでこの指示を使えます。コピーだけではログインも送信もしません。':'Use the prompt in your official ChatGPT, Codex or Claude app. Copying does not sign in or send a request.'}</p><button className="cr-primary" onClick={()=>void copy()}>{ja?'指示をコピー':'Copy instructions'}</button><a href="#install" onClick={onClose}>{ja?'ローカルCLIを設定':'Set up the local CLI'}</a></section>}
    {route==='api'&&<section className="cr-form">
      <label>Provider<select value={provider} onChange={e=>{setProvider(e.target.value);setBase(e.target.value==='ollama'?'http://127.0.0.1:11434/':'https://api.openai.com/v1');setKey('');setConsent(false);}}><option value="compatible">OpenAI-compatible API</option><option value="ollama">Ollama / local</option></select></label>
      <label>Endpoint<input type="url" value={base} autoComplete="off" onChange={e=>{setBase(e.target.value);setKey('');setConsent(false);}}/></label>
      <label>Model ID<input value={model} autoComplete="off" placeholder="Your available model" onChange={e=>{setModel(e.target.value);setConsent(false);}}/></label>
      <label>API key / optional<input type="password" value={key} autoComplete="off" spellCheck={false} onChange={e=>setKey(e.target.value)}/></label>
      <p>{ja?'キーはこのタブのメモリだけに置き、送信開始または画面を閉じる時に消します。指定先にはキーと依頼が渡ります。提供先の料金・保持方針が適用されます。CORSやローカルネットワーク制限で接続できない場合があります。':'The key stays in tab memory and is cleared when sending starts or this dialog closes. The chosen endpoint receives the key and prompt. Its billing and retention policies apply. Browser CORS or private-network rules may block access.'}</p>
      <label className="cr-check"><input type="checkbox" checked={consent} disabled={busy} onChange={e=>setConsent(e.target.checked)}/>{ja?'上記の接続先へ、この内容を送信する':'Send this content to the endpoint displayed above'}</label>
      <button className="cr-primary" disabled={busy||!consent} onClick={()=>void call()}>{busy?(ja?'応答待ち':'Waiting'):(ja?'この接続先へ送信':'Send to this endpoint')}</button>
      {busy&&<button onClick={()=>controller.current?.abort()}>{ja?'キャンセル':'Cancel'}</button>}
    </section>}
    {route==='import'&&<section><label>{ja?'外部AIの回答（JSONまたは文章）':'External AI answer (JSON or text)'}<textarea rows={6} value={raw} maxLength={80000} onChange={e=>setRaw(e.target.value)}/></label><p>{ja?'コードは実行しません。出典は本人の手動取り込みとして残します。':'No code is executed. Provenance records a manual import, not an authenticated model receipt.'}</p><button disabled={!raw.trim()} onClick={()=>{try{receive(raw,'Manual import · unverified · '+new Date().toISOString());setNotice(ja?'未確認の回答として取り込みました。':'Imported as an unverified answer.');setRaw('');}catch{setNotice('No answer to import.');}}}>{ja?'候補として取り込む':'Import as a proposal'}</button></section>}
    <output role="status">{notice}</output>
  </dialog>;
}
