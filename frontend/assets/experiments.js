// The style and chat exercises share the guided lab's helpers and visual language.
// Each experiment locks only its own controls; reading and local math remain usable.
(()=>{
'use strict';
const names={novels:'Fiction',lyrics:'Lyrics',news:'News',academic:'Academic'};
const tokenKey=t=>JSON.stringify(t.bytes??t.token);
let styleBusy=false,chatBusy=false,history=[];
function lock(root,value){for(const control of root.querySelectorAll('button,input,textarea,select'))control.disabled=value||control.dataset.permanentDisabled==='true';}
function resultPanel(parent,style){const article=make('article','result-panel');article.append(make('h3','',names[style]));const loading=make('p','fine','Waiting for the model…');article.append(loading);parent.append(article);return {article,loading};}
function comparison(targetId,results,selected,title){
 const good=results.filter(Boolean),target=$(targetId);target.replaceChildren();if(!good.length)return;
 target.append(make('h3','comparison-title',title),make('p','fine','Each row is the same exact token, including spaces. “Not returned” means outside that response’s returned candidates, not zero. Probabilities are not renormalized.'));
 const maps=new Map(good.map(r=>[r.style,new Map(r.options.map(t=>[tokenKey(t),t]))]));
 const union=new Map();for(const result of good)for(const t of result.options){const key=tokenKey(t);const previous=union.get(key);union.set(key,{token:t.token,key,largest:Math.max(previous?.largest??0,t.probability)});}
 const ranked=[...union.values()].sort((a,b)=>b.largest-a.largest);
 const wrapper=make('div','table-scroll comparison-scroll');const table=make('table','numeric-table style-table');const head=make('thead'),heading=make('tr');for(const text of ['Token',...selected.map(style=>names[style])]){const th=make('th','',text);th.scope='col';heading.append(th);}head.append(heading);table.append(head);const body=make('tbody'),hidden=[];
 for(const [index,item]of ranked.entries()){
  const row=make('tr');row.append(make('td','',JSON.stringify(item.token)));
  for(const style of selected){const option=maps.get(style)?.get(item.key),cell=make('td','',option?percent(option.probability):maps.has(style)?'Not returned':'Request failed');if(option){const track=make('div','prob-track'),fill=make('div','prob-fill');fill.style.width=(option.probability*100)+'%';track.append(fill);cell.append(track);}row.append(cell);}
  if(index>=8){row.hidden=true;hidden.push(row);}body.append(row);
 }
 const other=make('tr');other.append(make('td','','Other (outside top 10)'));for(const style of selected){const result=good.find(r=>r.style===style);other.append(make('td','',result?percent(result.other):'Request failed'));}body.append(other);table.append(body);wrapper.append(table);target.append(wrapper);
 if(hidden.length){const toggle=make('button','text-button',`Show ${hidden.length} more tokens`);toggle.type='button';toggle.setAttribute('aria-expanded','false');toggle.addEventListener('click',()=>{const expanded=toggle.getAttribute('aria-expanded')!=='true';for(const row of hidden)row.hidden=!expanded;toggle.setAttribute('aria-expanded',String(expanded));toggle.textContent=expanded?'Show fewer tokens':`Show ${hidden.length} more tokens`;});target.append(toggle);}
}
function clearStyles(message){$('style-comparison').replaceChildren();$('style-results').replaceChildren();$('style-snapshot').hidden=true;status('style-status',message);}
for(const id of ['style-prefix','style-selection','style-temperature'])$(id).addEventListener('input',()=>{if(id==='style-temperature')$('style-temp-value').value=Number($('style-temperature').value).toFixed(1);clearStyles('Inputs changed. Compare again to request fresh distributions.');});
$('copy-prefix').addEventListener('click',()=>{$('style-prefix').value=$('prefix').value;clearStyles('Prefix copied from section 6. Compare styles to request new probabilities.');});
function renderStyle(article,data,prefix){
 article.append(make('p','result-meta',data.model));const sample=make('p','sampled-token');sample.append(make('span','','Sampled token: '),make('code','',JSON.stringify(data.generated_token)));article.append(sample);
 const detail=make('details'),summary=make('summary','','Inspect this style’s top 10');detail.append(summary);for(const t of data.tokens)detail.append(bar(JSON.stringify(t.token),t.probability));detail.append(bar('Other',data.other_probability,'other'));article.append(detail);
 const append=make('button','','Append this token to the shared prefix');append.type='button';append.disabled=true;let text=data.generated_token;
 try{if(data.generated_token_bytes)text=new TextDecoder('utf-8',{fatal:true}).decode(new Uint8Array(data.generated_token_bytes));}catch{append.dataset.permanentDisabled='true';append.textContent='Partial UTF-8 token: cannot append by itself';}
 append.addEventListener('click',()=>{if(styleBusy)return;if($('style-prefix').value!==prefix){status('style-status','The prefix changed. Compare again before appending.',true);return;}if(prefix.length+text.length>4000){status('style-status','The updated prefix would exceed 4,000 characters.',true);return;}$('style-prefix').value=prefix+text;clearStyles('One token appended to the shared prefix. Compare again; every style will receive this updated text.');});article.append(append);
}
$('style-form').addEventListener('submit',async event=>{
 event.preventDefault();if(styleBusy)return;const prefix=$('style-prefix').value;if(!prefix.trim()){status('style-status','Enter an unfinished sentence first.',true);return;}
 const temperature=Number($('style-temperature').value),selected=$('style-selection').value==='all'?Object.keys(names):[$('style-selection').value];styleBusy=true;clearStyles('Requesting one distribution per selected style…');lock($('styles'),true);$('style-snapshot').hidden=false;$('style-snapshot').textContent=`Same text: ${JSON.stringify(prefix)} · Temperature ${temperature.toFixed(1)} · ${selected.length} API request${selected.length===1?'':'s'}`;
 try{const results=await Promise.all(selected.map(async style=>{const {article,loading}=resultPanel($('style-results'),style);try{const data=await api('/next-token',{prompt:prefix,corpus:style,temperature,top_logprobs:10});loading.remove();renderStyle(article,data,prefix);return {style,options:data.tokens,other:data.other_probability};}catch(error){loading.remove();article.append(make('p','status error',error.message));return null;}}));comparison('style-comparison',results,selected,'Compare the same next-token candidates');const count=results.filter(Boolean).length;status('style-status',`${count} of ${selected.length} styles loaded. ${count?'Compare the rows, or append a sampled token and compare again.':'Check the request errors below.'}`,count===0);}
 finally{styleBusy=false;lock($('styles'),false);}
});
function showHistory(){const root=$('chat-history');root.replaceChildren();if(!history.length){root.append(make('p','fine','No reply chosen yet. All styles start with the same empty history.'));return;}for(const message of history){const turn=make('div','history-turn');turn.append(make('strong','',message.role==='user'?'You':'Chosen assistant reply'),make('p','',message.content));root.append(turn);}}
function renderReply(article,data,messages){
 article.append(make('p','result-meta',data.model+' · Temperature '+data.temperature),make('div','reply',data.text));if(data.finish_reason==='length')article.append(make('p','fine','This reply reached the output limit and may end mid-sentence.'));
 const detail=make('details','reply-inspector');detail.append(make('summary','',`Inspect ${data.positions.length} generated tokens`),make('p','fine','Select a chip. The inspector shows the probability and alternatives at that token’s exact position, conditioned on the previous output in this reply.'));
 const chips=make('div','reply-tokens'),inspector=make('div','token-inspector');
 data.positions.forEach((position,index)=>{const button=make('button','token-button',JSON.stringify(position.token));button.type='button';button.setAttribute('aria-pressed','false');button.setAttribute('aria-label',`Inspect ${names[data.corpus]} token ${index+1}: ${JSON.stringify(position.token)}`);button.addEventListener('click',()=>{
  for(const sibling of chips.children)sibling.setAttribute('aria-pressed',String(sibling===button));
  inspector.replaceChildren(make('h4','',`Token ${index+1}: ${JSON.stringify(position.token)}`),make('p','',`${percent(position.probability)} probability · logprob ${position.logprob.toFixed(3)}`),make('p','inspector-context',index===0?'Conditioning: shared conversation plus this style’s instruction.':'Previous output in this reply: '+JSON.stringify(data.positions.slice(0,index).map(p=>p.token).join(''))));
  for(const option of position.alternatives)inspector.append(bar(JSON.stringify(option.token),option.probability));inspector.append(bar('Other',position.other_probability,'other'));
 });chips.append(button);});detail.append(chips,inspector);article.append(detail);if(chips.firstChild)chips.firstChild.click();
 const keep=make('button','','Continue with this reply');keep.type='button';keep.disabled=true;keep.addEventListener('click',()=>{if(chatBusy)return;history=[...messages,{role:'assistant',content:data.text}];showHistory();$('chat-results').replaceChildren();$('chat-first-comparison').replaceChildren();if($('chat-message').value===messages.at(-1).content)$('chat-message').value='';$('chat-message').focus();status('chat-status',`Kept the ${names[data.corpus].toLowerCase()} reply. Ask a follow-up; every style will receive the same shared history shown above.`);});article.append(keep);
}
$('chat-temperature').addEventListener('input',()=>{$('chat-temp-value').value=Number($('chat-temperature').value).toFixed(1);});
$('chat-form').addEventListener('submit',async event=>{
 event.preventDefault();if(chatBusy)return;const message=$('chat-message').value;if(!message.trim())return;const messages=[...history,{role:'user',content:message}],temperature=Number($('chat-temperature').value);
 if(messages.length>13||messages.reduce((n,m)=>n+m.content.length,0)>16000){status('chat-status','The shared conversation has reached the lab limit. Start a new chat or shorten your message.',true);return;}
 chatBusy=true;$('chat-results').replaceChildren();$('chat-first-comparison').replaceChildren();lock($('chat'),true);status('chat-status',`Generating four replies for ${JSON.stringify(message)} · Temperature ${temperature.toFixed(1)}. The shared history is unchanged until you choose a reply.`);
 try{const results=await Promise.all(Object.keys(names).map(async style=>{const {article,loading}=resultPanel($('chat-results'),style);try{const data=await api('/chat',{messages,corpus:style,temperature});if(!data.positions?.length)throw new Error('No token probabilities were returned for this reply.');loading.remove();renderReply(article,data,messages);return {style,options:data.positions[0].alternatives,other:data.positions[0].other_probability};}catch(error){loading.remove();article.append(make('p','status error',error.message));return null;}}));comparison('chat-first-comparison',results,Object.keys(names),'First output token: shared conversation, different style instructions');const count=results.filter(Boolean).length;status('chat-status',`${count} of 4 replies loaded. ${count?'Read the replies, open a token inspector, or choose a reply to keep the exchange for your next turn.':'Check the request errors below.'}`,count===0);}
 finally{chatBusy=false;lock($('chat'),false);}
});
$('new-chat').addEventListener('click',()=>{history=[];showHistory();$('chat-results').replaceChildren();$('chat-first-comparison').replaceChildren();$('chat-message').value='';status('chat-status','Shared history cleared. Enter a message to begin again.');});
})();
