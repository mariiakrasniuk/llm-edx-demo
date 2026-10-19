(()=>{
'use strict';
const links=[...document.querySelectorAll('.outline-link')];
const outline=document.querySelector('.course-outline');
if(window.matchMedia('(max-width:800px)').matches)outline.open=false;
function updateProgress(){
 const completed=[...document.querySelectorAll('.finish-step[aria-pressed="true"]')].map(b=>b.dataset.step);
 document.getElementById('course-progress').value=completed.length;
 document.getElementById('course-progress-label').textContent=`${completed.length} of 4 activities`;
 for(const link of links){const step=link.dataset.outlineStep;if(!step)continue;const done=completed.includes(step);link.classList.toggle('is-complete',done);link.querySelector('.outline-marker').textContent=done?'✓':step;}
}
const observer=new MutationObserver(updateProgress);
for(const button of document.querySelectorAll('.finish-step'))observer.observe(button,{attributes:true,attributeFilter:['aria-pressed']});
let pending=false;
function updateLocation(){
 pending=false;let current=links[0];
 for(const link of links){const target=document.getElementById(link.hash.slice(1));if(target&&target.getBoundingClientRect().top<180)current=link;}
 for(const link of links){if(link===current)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');}
}
window.addEventListener('scroll',()=>{if(!pending){pending=true;requestAnimationFrame(updateLocation);}},{passive:true});
for(const link of links)link.addEventListener('click',()=>{if(window.matchMedia('(max-width:800px)').matches)outline.open=false;});
updateProgress();updateLocation();
})();
