/* Local presentation only. No inference, analytics, fetches or storage. */
(() => {
  'use strict';
  const data = window.UDGAM_RECORDED;
  if (!data || !Array.isArray(data.examples) || data.examples.length !== 3) return;
  const byId = id => document.getElementById(id);
  const exampleButtons = [...document.querySelectorAll('[data-example]')];
  const viewButtons = [...document.querySelectorAll('[data-view]')];
  let selected = 0;
  let view = 'plain';

  function showExample(index, announce = true) {
    const item = data.examples[index];
    if (!item) return;
    selected = index;
    byId('example-text').textContent = item.text;
    byId('example-id').textContent = item.id;
    byId('example-outcome').textContent = item.outcome;
    byId('example-outcome').classList.toggle('failure', !item.success);
    document.querySelector('.example-stage').classList.toggle('is-failure', !item.success);
    byId('example-explanation').textContent = item.explanation;
    byId('example-intent').textContent = item.plain_intent;
    byId('example-application').textContent = item.application;
    byId('example-fields').replaceChildren(...item.fields.map(([label, value]) => {
      const row = document.createElement('div');
      const dt = document.createElement('dt');
      const dd = document.createElement('dd');
      dt.textContent = label;
      dd.textContent = value;
      row.append(dt, dd);
      return row;
    }));
    byId('example-json').textContent = JSON.stringify(item.tree, null, 2);
    byId('raw-candidate').textContent = item.prediction;
    byId('raw-base').textContent = item.base_prediction;
    byId('raw-reference').textContent = item.reference;
    exampleButtons.forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
    if (announce) byId('example-announcement').textContent = `Recorded example ${index + 1}: ${item.name}. ${item.outcome}.`;
  }

  function showView(next) {
    if (!['plain', 'json'].includes(next)) return;
    view = next;
    byId('plain-view').hidden = view !== 'plain';
    byId('json-view').hidden = view !== 'json';
    viewButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.view === view)));
  }

  // Native buttons work with Tab, Enter and Space. Arrow keys also move within a group.
  function moveInGroup(event, buttons, current, activate) {
    let next;
    if (event.key === 'ArrowRight') next = (current + 1) % buttons.length;
    else if (event.key === 'ArrowLeft') next = (current + buttons.length - 1) % buttons.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = buttons.length - 1;
    else return;
    event.preventDefault();
    buttons[next].focus();
    activate(next);
  }
  exampleButtons.forEach((button, index) => {
    button.addEventListener('click', () => showExample(index));
    button.addEventListener('keydown', event => moveInGroup(event, exampleButtons, index, showExample));
  });
  viewButtons.forEach((button, index) => {
    button.addEventListener('click', () => showView(button.dataset.view));
    button.addEventListener('keydown', event => moveInGroup(event, viewButtons, index, i => showView(viewButtons[i].dataset.view)));
  });
  showExample(selected, false);
  showView(view);
})();
