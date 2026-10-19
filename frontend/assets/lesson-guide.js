(()=>{
'use strict';
const completed=new Set();
for(const button of document.querySelectorAll('.finish-step'))button.addEventListener('click',()=>{
 const step=button.dataset.step;
 if(completed.has(step))completed.delete(step);else completed.add(step);
 button.setAttribute('aria-pressed',String(completed.has(step)));
 button.textContent=completed.has(step)?'Done ✓':'I’ve tried this ✓';
 document.getElementById('journey-progress').textContent=completed.size===4?'All four experiments explored. Try the quick check below, or choose an optional extra.':`${completed.size} of 4 experiments explored. Go at your own pace.`;
 document.querySelectorAll('.journey-nav a').forEach((link,i)=>{link.classList.toggle('completed',completed.has(String(i+1)));});
});
const feedback={training:'Not quite. Training changes the model’s numerical settings using examples. Editing your prompt leaves those settings alone. Try another answer.',context:'Exactly. The input context changes. The same trained settings can produce different predictions from different clues.',truth:'A changed prompt can change predictions, but it does not guarantee truth. A model can still produce a convincing mistake. Try another answer.'};
for(const button of document.querySelectorAll('[data-check]'))button.addEventListener('click',()=>{
 document.getElementById('check-feedback').textContent=feedback[button.dataset.check];
 for(const other of document.querySelectorAll('[data-check]'))other.setAttribute('aria-pressed',String(other===button));
});
document.getElementById('q-prompt').addEventListener('input',()=>{document.getElementById('q-status').textContent='Text changed. Click “Show me the pieces” to update the experiments below.';});
// Linked optional tools also open when reached with an anchor.
function revealAnchor(){const node=document.getElementById(location.hash.slice(1));if(!node)return;let ancestor=node.parentElement;while(ancestor){if(ancestor.tagName==='DETAILS')ancestor.open=true;ancestor=ancestor.parentElement;}}
window.addEventListener('hashchange',revealAnchor);revealAnchor();
})();

(()=>{
'use strict';
const get=id=>document.getElementById(id);
const examples=['She sat beside the river bank.','She deposited money at the bank.'];
const question="What does the word 'bank' mean in this sentence? Explain in one short sentence.";
let pending=false;
function clear(){get('meaning-results').replaceChildren();get('meaning-takeaway').hidden=true;get('meaning-status').textContent='Sentences changed. Make a prediction, then ask the model again.';}
for(const id of ['meaning-a','meaning-b'])get(id).addEventListener('input',clear);
get('meaning-control').onclick=()=>{get('meaning-b').value=get('meaning-a').value;clear();get('meaning-status').textContent='Both sentences are now identical. Predict whether the answers will match.';};
get('meaning-reset').onclick=()=>{get('meaning-a').value=examples[0];get('meaning-b').value=examples[1];clear();};
get('meaning-form').onsubmit=async event=>{
 event.preventDefault();if(pending)return;
 const sentences=[get('meaning-a').value,get('meaning-b').value];
 if(sentences.some(s=>!s.trim())){get('meaning-status').textContent='Please enter a sentence in both boxes.';return;}
 pending=true;get('meaning-form').querySelectorAll('button,textarea').forEach(n=>n.disabled=true);get('meaning-results').replaceChildren();get('meaning-takeaway').hidden=true;get('meaning-status').textContent='Asking Qwen the same question about each sentence…';
 try{
  const results=await Promise.all(sentences.map(async(sentence,i)=>{
   const card=document.createElement('article');card.className='meaning-answer';const heading=document.createElement('h3');heading.textContent=`${i?'B':'A'} · ${sentence}`;const reply=document.createElement('p');reply.textContent='Waiting for the model…';card.append(heading,reply);get('meaning-results').append(card);
   try{const result=await api('/chat',{messages:[{role:'user',content:question+'\nSentence: '+sentence}],corpus:'academic',temperature:0});reply.textContent=result.text;const source=document.createElement('p');source.className='fine';source.textContent=result.model+' · same instruction · most-likely-token selection';card.append(source);return true;}
   catch(error){reply.textContent=error.message;reply.className='error';return false;}
  }));
  const count=results.filter(Boolean).length;get('meaning-status').textContent=count===2?'Both answers are ready. Compare them with your prediction. Only the supplied sentence changed.':`${count} of 2 answers loaded. Try again to complete the comparison.`;get('meaning-takeaway').hidden=count!==2;get('meaning-observation').textContent=sentences[0]===sentences[1]?'With identical sentences, the inputs and selection settings match. The answers should match too; compare what you see. No retraining happened.':'Compare each answer with your prediction. If the meanings differ appropriately, this shows that changing the context can change the result without retraining. Unexpected answers are worth discussing: the model can make mistakes.';
 }finally{pending=false;get('meaning-form').querySelectorAll('button,textarea').forEach(n=>n.disabled=false);}
};
})();
