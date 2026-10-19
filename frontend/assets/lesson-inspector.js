(()=>{
'use strict';
const $=id=>document.getElementById('q-'+id);
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
let data=null,selected=0,history=[],busy=false,activeHeads=[0],stage='block_input';
const pct=p=>(100*p).toFixed(2)+'%';
let requestController=null;
async function post(path,body){
 const controller=new AbortController();requestController=controller;
 let timedOut=false;
 const timeout=setTimeout(()=>{timedOut=true;controller.abort();},30000);
 const cancel=$('cancel');cancel.hidden=false;cancel.disabled=false;$('prediction-cancel').hidden=false;$('prediction-cancel').disabled=false;
 try{
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:controller.signal});
  if(!r.ok){let message;try{const e=await r.json();message=typeof e.detail==='string'?e.detail:JSON.stringify(e.detail);}catch{message=`Server error (${r.status})`;}throw Error(message);}
  return await r.json();
 }catch(error){
  if(error.name==='AbortError')throw Error(timedOut?'No response reached this page within 30 seconds. Previous results are unchanged. Check VS Code’s forwarded port 8000, then try again.':'Request cancelled. Previous results are unchanged.');
  if(error instanceof TypeError)throw Error('Connection lost. Check VS Code’s forwarded port 8000, then try again. Previous results are unchanged.');
  throw error;
 }finally{clearTimeout(timeout);if(requestController===controller){requestController=null;cancel.hidden=true;$('prediction-cancel').hidden=true;}}
}
$('cancel').onclick=$('prediction-cancel').onclick=()=>requestController?.abort();
function inspectionStatus(message,error=false){for(const id of ['status','prediction-status']){$(id).textContent=message;$(id).classList.toggle('error',error);}}

function lock(value){busy=value;document.querySelectorAll('#qwen-lessons button,#qwen-lessons select,#qwen-lessons textarea,#qwen-lessons input,.inspector-extra button,.inspector-extra select,.inspector-extra textarea,.inspector-extra input').forEach(n=>n.disabled=value);if(!value){$('undo').disabled=!history.length;$('sample').disabled=$('greedy').disabled=!data||data.tokens.length>=96;$('download').disabled=!data;}}
async function inspect(ids,reset=false){if(busy)return;lock(true);inspectionStatus('Waiting for Qwen… Previous results stay visible until this request finishes.');try{const next=await post('/inspect',{text:$('prompt').value,token_ids:ids,layer:Number($('layer').value||0),head:activeHeads[0]??0,temperature:Number($('temperature').value),query_delta:Number($('delta').value)});if(reset)history=[];data=next;selected=Math.min(selected,data.tokens.length-1);render();inspectionStatus(`${data.model} · ${data.tokens.length} tokens · Real forward pass. Ready: choose Sample and append or Append most likely.`);}catch(e){inspectionStatus(e.message,true);}finally{lock(false);}}
function options(id,count,current){const s=$(id);s.replaceChildren();for(let i=0;i<count;i++){const o=el('option',String(i+1));o.value=i;s.append(o);}s.value=current;}
function render(){
 $('current-text').textContent='Text so far: '+data.text;
 $('token-count').textContent=`${data.text.trim().split(/\s+/).filter(Boolean).length} whitespace-separated words · ${data.tokens.length} Qwen tokens`;$('pass-label').textContent=data.query_delta?`Experimental pass: Q weight change ${data.query_delta}; server weights restored after inspection.`:'Original trained weights.';$('tokens').replaceChildren();data.tokens.forEach((t,i)=>{const n=el('span',`${i+1} · ${JSON.stringify(t.text)}`,'token');n.append(el('small',`ID ${t.id} · ${t.piece}`));n.tabIndex=0;n.setAttribute('role','button');n.onclick=()=>{selected=i;drawAttention();renderExtras();};n.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();n.click();}};$('tokens').append(n);});
 options('layer',data.layers,data.layer);activeHeads=activeHeads.filter(h=>h<data.heads);if(!activeHeads.length)activeHeads=[0];drawAttention();
 $('vector-note').textContent=`Each token has ${data.width} coordinates. Compare its learned input embedding with ${data.hidden_label.toLowerCase()} at layer ${data.layer+1}.`;
 const table=$('vectors');table.replaceChildren();const h=el('tr');['Token','Vector',...Array.from({length:8},(_,i)=>`d${i+1}`)].forEach(t=>h.append(el('th',t)));table.append(h);
 data.tokens.forEach((t,i)=>{for(const [name,values]of [['Embedding',data.embedding[i]],['Updated',data.hidden[i]]]){const r=el('tr');[JSON.stringify(t.text),name,...values.map(v=>v.toFixed(3))].forEach(v=>r.append(el('td',v)));table.append(r);}});
 $('probabilities').replaceChildren();for(const t of [...data.alternatives,{text:'Other vocabulary',probability:data.other_probability}]){const r=el('div',undefined,'prob');const track=el('div',undefined,'track'),fill=el('div',undefined,'fill');fill.style.width=100*t.probability+'%';track.append(fill);r.append(el('code',JSON.stringify(t.text)),track,el('span',pct(t.probability)));$('probabilities').append(r);}
 renderExtras();
}
function drawAttention(){
 const colors=Array.from({length:data.heads},(_,i)=>`hsl(${i*137.5%360} 60% 44%)`);
 const controls=$('head-controls');controls.replaceChildren();for(let h=0;h<data.heads;h++){const btn=el('button',String(h+1));btn.style.background=colors[h];btn.style.opacity=activeHeads.includes(h)?1:.3;btn.setAttribute('aria-pressed',String(activeHeads.includes(h)));btn.setAttribute('aria-label',`Toggle head ${h+1}`);btn.onclick=()=>{activeHeads=activeHeads.includes(h)?activeHeads.filter(i=>i!==h):[...activeHeads,h];if(!activeHeads.length)activeHeads=[h];drawAttention();};controls.append(btn);}
 const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 560 ${data.tokens.length*32+24}`);svg.setAttribute('aria-label','Attention head connections');
 function node(tag,attrs,text){const n=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
 for(const h of activeHeads)data.all_heads[h][selected].forEach((p,j)=>{if(p>0)svg.append(node('path',{d:`M 175 ${selected*32+20} L 380 ${j*32+20}`,stroke:colors[h],'stroke-width':Math.max(.3,p*12),opacity:Math.max(.08,p*.85),fill:'none'}));});
 data.tokens.forEach((t,i)=>{const label=`${i+1} ${JSON.stringify(t.text)}`;const group=node('g',{tabindex:0,role:'button','aria-label':`Inspect token ${i+1}: ${t.text}`});group.append(node('rect',{x:0,y:i*32+3,width:175,height:30,fill:i===selected?'#e7e5ef':'transparent'}),node('text',{x:6,y:i*32+24},label.length>22?label.slice(0,21)+'…':label));const choose=()=>{if(selected===i)return;selected=i;drawAttention();renderExtras();};group.onclick=choose;group.onmouseenter=choose;group.onfocus=choose;group.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}};svg.append(group);const p=data.all_heads[activeHeads[0]][selected][i];svg.append(node('rect',{x:380,y:i*32+3,width:180,height:30,fill:colors[activeHeads[0]],opacity:p*.5}),node('text',{x:386,y:i*32+24},label.length>22?label.slice(0,21)+'…':label));});$('head-view').replaceChildren(svg);
 const table=el('table'),header=el('tr');header.append(el('th','Query ↓ / key →'));data.tokens.forEach((t,i)=>{const th=el('th',String(i+1));th.title=t.text;header.append(th);});table.append(header);
 data.all_heads[activeHeads[0]].forEach((row,i)=>{const tr=el('tr');tr.append(el('th',`${i+1} ${data.tokens[i].text}`));row.forEach((p,j)=>{const td=el('td');td.style.background=j>i?'#eee':`rgba(84,59,150,${p*.7})`;const b=el('button',j>i?'—':(p*100).toFixed(0));b.title=`Token ${i+1} → ${j+1}: ${pct(p)}`;b.setAttribute('aria-label',b.title);b.onclick=()=>{selected=i;drawAttention();renderExtras();$('selection').textContent+=` Connection ${i+1} → ${j+1}: ${pct(p)}.`;};td.append(b);tr.append(td);});table.append(tr);});$('heatmap').replaceChildren(table);$('selection').textContent=`Query token ${selected+1}: ${JSON.stringify(data.tokens[selected].text)} · Layer ${data.layer+1}; heads ${activeHeads.map(h=>h+1).join(', ')}. Heatmap: head ${activeHeads[0]+1}.`;
}
$('prediction-form').onsubmit=e=>{e.preventDefault();if(busy)return;$('prompt').value=$('prediction-input').value;$('delta').value=0;$('delta-value').value='0.00';inspect(undefined,true);};
$('prediction-input').addEventListener('input',()=>{$('prediction-status').textContent='Draft changed. Click Start this sentence to replace the current text and predictions.';});
$('inspect-form').onsubmit=e=>{e.preventDefault();inspect(undefined,true);};
$('example').onclick=()=>{$('prompt').value='The trophy would not fit in the suitcase because it was too big.';inspect(undefined,true);};
for(const id of ['layer','temperature'])$(id).onchange=()=>{if(data)inspect(data.tokens.map(t=>t.id));};
async function append(id){if(!data||busy)return;if(data.eos_ids.includes(id)){inspectionStatus('Qwen selected an end-of-sequence token. Start a new sentence to continue.');return;}const previous=data;await inspect([...data.tokens.map(t=>t.id),id]);if(data!==previous){history.push(previous);$('undo').disabled=false;}}
$('sample').onclick=()=>{if(data)append(data.sampled_id);};$('greedy').onclick=()=>{if(data)append(data.greedy_id);};
$('undo').onclick=()=>{if(history.length){data=history.pop();$('temperature').value=String(data.temperature);$('delta').value=data.query_delta;$('delta-value').value=data.query_delta.toFixed(2);render();lock(false);}};
$('download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download='qwen-evidence.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};

function renderExtras(){
 const names={embedding:'Token embedding',block_input:'Block input',attention_update:'Attention update',after_attention:'Add residual',mlp_update:'MLP update',hidden:data.hidden_label};
 const flow=$('flow');flow.replaceChildren();for(const [id,name]of Object.entries(names)){const b=el('button',name);b.setAttribute('aria-pressed',String(stage===id));b.onclick=()=>{stage=id;renderExtras();};flow.append(b);}
 const values=data[stage]??data.stages[stage];$('stage-values').replaceChildren(el('h3',`${names[stage]} · token ${selected+1} ${JSON.stringify(data.tokens[selected].text)}`));const strip=el('div',undefined,'coordinates');values[selected].forEach(v=>{const c=el('span',v.toFixed(4));c.style.background=v>=0?'#e4effa':'#fae9df';strip.append(c);});$('stage-values').append(strip);
 $('temperatures').replaceChildren();for(const d of data.temperature_comparison){const panel=el('article',undefined,'result-panel');panel.append(el('h3',`Temperature ${d.temperature}`));d.alternatives.slice(0,5).forEach((t,i)=>panel.append(el('p',`${JSON.stringify(t.token)}: ${pct(t.probability)} · ${d.counts[i]} / 100 draws`)));panel.append(el('p',`Remaining probability: ${pct(1-d.alternatives.slice(0,5).reduce((a,t)=>a+t.probability,0))}`));$('temperatures').append(panel);}
 $('window-size').max=data.tokens.length;
}
$('all-heads').onclick=()=>{if(data){activeHeads=Array.from({length:data.heads},(_,i)=>i);drawAttention();}};
$('one-head').onclick=()=>{activeHeads=[activeHeads[0]??0];if(data)drawAttention();};
$('delta').oninput=()=>{$('delta-value').value=Number($('delta').value).toFixed(2);};
$('apply-weight').onclick=async()=>{if(!data||busy)return;const before=data;const ids=data.tokens.map(t=>t.id);lock(true);try{const base=await post('/inspect',{token_ids:ids,layer:data.layer,head:activeHeads[0],query_delta:0,temperature:Number($('temperature').value)});const changed=await post('/inspect',{token_ids:ids,layer:data.layer,head:activeHeads[0],query_delta:Number($('delta').value),temperature:Number($('temperature').value)});data=changed;render();const difference=Math.max(...base.attention.flat().map((x,i)=>Math.abs(x-changed.attention.flat()[i])));$('weight-result').textContent=`Weight ${base.query_weight.toFixed(5)} → ${(base.query_weight+changed.query_delta).toFixed(5)}. Largest attention change in head ${activeHeads[0]+1}: ${(difference*100).toFixed(4)} percentage points. The original weight has been restored on the server. The displays show the experimental pass.`;}catch(e){data=before;$('weight-result').textContent=e.message;}finally{lock(false);}};
$('reset-weight').onclick=()=>{$('delta').value=0;$('delta-value').value='0.00';$('weight-result').textContent='Original trained weight selected.';if(data)inspect(data.tokens.map(t=>t.id));};
$('resample').onclick=()=>{if(data)inspect(data.tokens.map(t=>t.id));};
$('window-run').onclick=async()=>{if(!data||busy)return;const keep=Number($('window-size').value);if(!Number.isInteger(keep)||keep<1||keep>data.tokens.length){$('window-status').textContent='Choose a valid token count.';return;}lock(true);try{const body={layer:data.layer,head:activeHeads[0],temperature:1,query_delta:0};const full=await post('/inspect',{...body,token_ids:data.tokens.map(t=>t.id)});const short=await post('/inspect',{...body,token_ids:data.tokens.slice(-keep).map(t=>t.id)});$('window-results').replaceChildren();for(const [name,r]of [['Full context',full],['Shortened context',short]]){const panel=el('article',undefined,'result-panel');panel.append(el('h3',name),el('p',JSON.stringify(r.text)));r.alternatives.forEach(t=>panel.append(el('p',`${JSON.stringify(t.text)}: ${pct(t.probability)}`)));$('window-results').append(panel);}$('window-status').textContent=`Compared ${full.tokens.length} tokens with ${keep}; unchanged trained weights, temperature 1.`;}catch(e){$('window-status').textContent=e.message;}finally{lock(false);}};
for(const button of document.querySelectorAll('[data-q-example]'))button.onclick=()=>{$('prompt').value=button.dataset.qExample;inspect(undefined,true);};
lock(false);
})();
