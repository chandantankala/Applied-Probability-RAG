const $ = id => document.getElementById(id);
let state = null, interaction = 'search', working = false;
let thread = sessionStorage.getItem('mathpaper-thread') || crypto.randomUUID();
sessionStorage.setItem('mathpaper-thread', thread);
function element(tag, cls, text) { const node=document.createElement(tag); if(cls) node.className=cls; if(text!==undefined) node.textContent=text; return node; }
function toast(text) { $('toast').textContent=text; $('toast').hidden=false; setTimeout(()=>$('toast').hidden=true,5000); }
async function api(url, payload) {
  const response = await fetch(url, payload ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)} : {});
  const data = await response.json();
  if(!response.ok) throw new Error(typeof data.detail==='string'?data.detail:'The request could not be completed.');
  return data;
}
function selectedPapers() { return [...document.querySelectorAll('.paper-option input:checked')].map(x=>x.value); }
async function refresh() {
  state = await api('/api/status');
  const before=new Set(selectedPapers()), first=!$('papers').querySelector('input');
  $('papers').replaceChildren();
  state.papers.forEach(p=>{
    const label=element('label','paper-option');
    const checkbox=document.createElement('input'); checkbox.type='checkbox'; checkbox.value=p.id; checkbox.checked=first||before.has(p.id);
    const detail=element('div'); detail.append(element('strong','',p.title),element('small','',`${p.pages} PDF pages · ${p.chunks} passages`));
    if(p.notes.toLowerCase().includes('review draft')) detail.append(element('small','paper-note','Uploaded review draft'));
    label.append(checkbox,detail); $('papers').append(label);
  });
  $('paper-count').textContent=state.papers.length;
  $('corpus-status').textContent=`${state.pages} pages · ${state.passages} indexed passages`;
  const notice=$('connection-notice');
  $('pdf-upload').closest('label').hidden=state.can_upload===false;
  $('candidates').max=state.max_candidates;
  $('candidates').value=Math.min(state.default_candidates,state.max_candidates);
  if(!state.models_ready || !state.index_ready) {
    notice.textContent='Model setup is needed for hybrid search. Run “mathpaper models” and “mathpaper index”. BM25 keyword search remains available after indexing.';
    notice.hidden=false;
  } else if(state.public_demo&&!state.chat_configured) {
    notice.textContent='This public demo supports evidence search across the included papers. Generated answers are not enabled yet.';
    notice.hidden=false;
  } else if(!state.chat_configured) {
    notice.textContent='Evidence search is ready. To generate agent answers, configure a tool-calling chat model and start the MCP server. See the project README.';
    notice.hidden=false;
  } else { notice.hidden=true; }
}
function addMessage(role,text) {
  $('welcome').hidden=true;
  const node=element('div','message '+role);
  if(role==='assistant') node.append(element('div','message-title','MathPaper'));
  node.append(element('p','',text)); $('messages').append(node); node.scrollIntoView({behavior:'smooth',block:'nearest'}); return node;
}
function showSources(sources) {
  $('evidence').replaceChildren(); $('source-count').textContent=sources.length; $('evidence-empty').hidden=!!sources.length;
  sources.forEach((source,i)=>{
    const card=element('article','source');
    const heading=element('div','source-top'); heading.append(element('span','source-number',String(i+1).padStart(2,'0')),element('span','',`PDF page ${source.page}`));
    card.append(heading,element('h3','',source.title));
    if(source.label) card.append(element('div','source-label',source.label));
    card.append(element('div','source-text',source.text));
    const link=element('a','',`Open original page ↗`); link.href=source.url; link.target='_blank'; link.rel='noopener';card.append(link);
    $('evidence').append(card);
  });
}
function citations(node, ids, sources) {
  const strip=element('div','citations');
  ids.forEach(id=>{const source=sources.find(s=>s.source_id===id);if(!source)return;const a=element('a','citation',`${source.label||'Source'} · p. ${source.page}`);a.href=source.url;a.target='_blank';a.rel='noopener';strip.append(a);});node.append(strip);
}
function setWorking(value) {working=value;$('busy').hidden=!value;$('send').disabled=value;}
$('composer').addEventListener('submit',async event=>{
  event.preventDefault(); if(working)return;
  const question=$('question').value.trim();if(!question)return;
  const paper_ids=selectedPapers();if(!paper_ids.length){toast('Select at least one paper.');return;}
  if(interaction==='chat'&&!state.chat_configured){toast('Configure the chat model and MCP server first. Find evidence is ready to use.');return;}
  addMessage('user',question);$('question').value='';setWorking(true);
  $('busy-label').textContent=interaction==='chat'?'The agent is gathering and checking evidence…':'Searching and ranking passages…';
  try {
    if(interaction==='search') {
      const result=await api('/api/search',{question,paper_ids,candidates:Number($('candidates').value),top_k:Number($('top-k').value),mode:$('retrieval-mode').value,max_length:512});
      showSources(result.sources);
      const message=addMessage('assistant',result.sources.length?`Found ${result.sources.length} passages relevant to your question. Inspect the evidence cards and open the original pages to check the mathematical details.`:'No passages were found for the selected papers. Try different wording.');
      message.append(element('small','',`${result.mode} · ${result.timing_ms} ms · ${result.candidates_scored} pairs reranked`));
      $('trace').textContent=JSON.stringify({mode:result.mode,candidates_scored:result.candidates_scored,timing_ms:result.timing_ms,score_note:result.score_note},null,2);
    } else {
      const result=await api('/api/chat',{question,paper_ids,thread_id:thread});showSources(result.sources);
      const node=addMessage('assistant',result.abstained?'I could not substantiate an answer from the selected papers within the search budget. Try naming a theorem or paper, or inspect the evidence directly.':'');
      result.claims.forEach(claim=>{node.append(element('p','',claim.text));citations(node,claim.source_ids,result.sources);});
      node.append(element('small','',`${result.tool_calls} tool calls · ${(result.timing_ms/1000).toFixed(1)} s · ${result.notice}`));
      $('trace').textContent=JSON.stringify(result.trace,null,2);
    }
    $('trace-details').hidden=false;
  } catch(error) {addMessage('assistant',error.message);}
  finally{setWorking(false);}
});
$('question').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();$('composer').requestSubmit();}});
document.querySelectorAll('[data-question]').forEach(button=>button.addEventListener('click',()=>{$('question').value=button.dataset.question;$('composer').requestSubmit();}));
document.querySelectorAll('[data-mode]').forEach(button=>button.addEventListener('click',()=>{interaction=button.dataset.mode;document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b===button));$('mode-caption').textContent=interaction==='chat'?'Research tools · Source checks · Cited answers':'Hybrid retrieval · MiniLM reranking';}));
$('new-chat').addEventListener('click',()=>{if(working)return;thread=crypto.randomUUID();sessionStorage.setItem('mathpaper-thread',thread);$('messages').replaceChildren();$('welcome').hidden=false;showSources([]);$('trace-details').hidden=true;});
$('pdf-upload').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;toast('Adding the paper and updating its index…');const data=new FormData();data.append('file',file);try{const response=await fetch('/api/upload',{method:'POST',body:data});const result=await response.json();if(!response.ok)throw new Error(result.detail);await refresh();toast(result.already_ingested?'This exact PDF is already in the collection.':'Paper added.');}catch(error){toast(error.message);}event.target.value='';});
refresh().catch(error=>{$('connection-notice').hidden=false;$('connection-notice').textContent='Could not connect to the Python service: '+error.message;});
