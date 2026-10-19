'use strict';
const $=id=>document.getElementById(id);
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
let data=null,selected=0,history=[],busy=false;
const pct=p=>(100*p).toFixed(2)+'%';
async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!r.ok){let message;try{const e=await r.json();message=typeof e.detail==='string'?e.detail:JSON.stringify(e.detail);}catch{message=`Server error (${r.status})`;}throw Error(message);}return r.json();}
function lock(value){busy=value;document.querySelectorAll('button,select,textarea').forEach(n=>n.disabled=value);if(!value){$('undo').disabled=!history.length;$('sample').disabled=$('greedy').disabled=!!data&&data.tokens.length>=96;}}
async function inspect(ids,reset=false){if(busy)return;lock(true);$('status').className='';$('status').textContent='Running Qwen…';try{const next=await post('/inspect',{text:$('prompt').value,token_ids:ids,layer:Number($('layer').value||0),head:Number($('head').value||0),temperature:Number($('temperature').value)});if(reset)history=[];data=next;selected=Math.min(selected,data.tokens.length-1);render();$('status').textContent=`${data.model} · ${data.tokens.length} tokens · Real forward pass`;}catch(e){$('status').textContent=e.message;$('status').className='error';}finally{lock(false);}}
function options(id,count,current){const s=$(id);s.replaceChildren();for(let i=0;i<count;i++){const o=el('option',String(i+1));o.value=i;s.append(o);}s.value=current;}
function render(){
 $('results').hidden=false;$('model').textContent=data.model;$('tokens').replaceChildren();data.tokens.forEach((t,i)=>{const n=el('span',`${i+1} · ${JSON.stringify(t.text)}`,'token');n.append(el('small',`ID ${t.id} · ${t.piece}`));$('tokens').append(n);});
 options('layer',data.layers,data.layer);options('head',data.heads,data.head);drawAttention();
 $('vector-note').textContent=`Each token has ${data.width} coordinates. Compare its learned input embedding with ${data.hidden_label.toLowerCase()} at layer ${data.layer+1}.`;
 const table=$('vectors');table.replaceChildren();const h=el('tr');['Token','Vector',...Array.from({length:8},(_,i)=>`d${i+1}`)].forEach(t=>h.append(el('th',t)));table.append(h);
 data.tokens.forEach((t,i)=>{for(const [name,values]of [['Embedding',data.embedding[i]],['Updated',data.hidden[i]]]){const r=el('tr');[JSON.stringify(t.text),name,...values.map(v=>v.toFixed(3))].forEach(v=>r.append(el('td',v)));table.append(r);}});
 $('probabilities').replaceChildren();for(const t of [...data.alternatives,{text:'Other vocabulary',probability:data.other_probability}]){const r=el('div',undefined,'prob');const track=el('div',undefined,'track'),fill=el('div',undefined,'fill');fill.style.width=100*t.probability+'%';track.append(fill);r.append(el('code',JSON.stringify(t.text)),track,el('span',pct(t.probability)));$('probabilities').append(r);}
}
function drawAttention(){
 const ns='http://www.w3.org/2000/svg';const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 460 ${data.tokens.length*30+24}`);svg.setAttribute('aria-label','Attention connections for selected token');
 function node(tag,attrs,text){const n=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
 data.attention[selected].forEach((p,j)=>{if(p>0)svg.append(node('path',{d:`M 140 ${selected*30+20} C 230 ${selected*30+20}, 230 ${j*30+20}, 320 ${j*30+20}`,stroke:'#19766b','stroke-width':Math.max(.25,p*14),opacity:.2+p*.8,fill:'none'}));});
 data.tokens.forEach((t,i)=>{const label=`${i+1} ${JSON.stringify(t.text)}`;const left=node('text',{x:4,y:i*30+24,tabindex:0,role:'button','aria-label':`Inspect token ${i+1}: ${t.text}`},label.length>17?label.slice(0,16)+'…':label);left.setAttribute('font-weight',i===selected?'bold':'normal');const choose=()=>{selected=i;drawAttention();};left.onclick=choose;left.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}};svg.append(left,node('text',{x:325,y:i*30+24},label.length>17?label.slice(0,16)+'…':label));});$('head-view').replaceChildren(svg);
 const table=el('table');const header=el('tr');header.append(el('th','↓ / →'));data.tokens.forEach((t,i)=>header.append(el('th',`${i+1}`)));table.append(header);
 data.attention.forEach((row,i)=>{const tr=el('tr');tr.append(el('th',`${i+1} ${data.tokens[i].text}`));row.forEach((p,j)=>{const td=el('td');td.style.background=j>i?'#f3f4f3':`rgba(25,118,107,${p*.8})`;const b=el('button',j>i?'—':(p*100).toFixed(0));b.title=`${i+1} → ${j+1}: ${pct(p)}`;b.setAttribute('aria-label',b.title);b.onclick=()=>{selected=i;drawAttention();$('selection').textContent+=` Selected connection: token ${i+1} → ${j+1}, ${pct(p)}.`;};td.append(b);tr.append(td);});table.append(tr);});$('heatmap').replaceChildren(table);$('selection').textContent=`Query token ${selected+1}: ${JSON.stringify(data.tokens[selected].text)} · Layer ${data.layer+1}, head ${data.head+1}.`;
}
$('inspect-form').onsubmit=e=>{e.preventDefault();inspect(undefined,true);};
$('example').onclick=()=>{$('prompt').value='The trophy would not fit in the suitcase because it was too big.';inspect(undefined,true);};
for(const id of ['layer','head','temperature'])$(id).onchange=()=>{if(data)inspect(data.tokens.map(t=>t.id));};
async function append(id){if(!data||busy)return;if(data.eos_ids.includes(id)){$('status').textContent='Qwen selected an end-of-sequence token. Start a new text to continue.';return;}const previous=data;await inspect([...data.tokens.map(t=>t.id),id]);if(data!==previous){history.push(previous);$('undo').disabled=false;}}
$('sample').onclick=()=>append(data.sampled_id);$('greedy').onclick=()=>append(data.greedy_id);
$('undo').onclick=()=>{if(history.length){data=history.pop();$('temperature').value=String(data.temperature);render();lock(false);}};
$('download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download='qwen-evidence.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('chat-form').onsubmit=async e=>{e.preventDefault();if(busy)return;lock(true);$('chat-status').textContent='Generating with the same Qwen model…';$('reply').textContent='';try{const r=await post('/reply',{text:$('chat').value});$('reply').textContent=r.text;$('chat-status').textContent=`${r.model} · ${r.note}`;}catch(err){$('chat-status').textContent=err.message;}finally{lock(false);}};
fetch('/health').then(r=>r.json()).then(r=>{$('model').textContent=r.model;}).catch(()=>{$('status').textContent='Cannot reach the backend.';});
