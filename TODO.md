# UI Overhaul TODO

## 1. Update static/css/style.css
- [x] Add modal styles (.modal, .modal-content, @keyframes zoomIn, fadeIn, fadeInRight)
- [x] Add toast styles (.toast, .toast.show)
- [x] Add agent preview styles (#agent-preview, #call-agent-preview)
- [x] Add disabled button styles (button:disabled)
- [x] Add tab styles for strategy modal (.tabs, .tab, .tab.active)
- [x] Add confirmation modal warning styles

## 2. Update templates/dashboard.html
- [x] Add agent dropdown to #panel-strategy (id="strategy-agent-select", required)
- [x] Add agent preview div below dropdown (#agent-preview)
- [x] Replace "Ingest Strategy" button with one opening modal
- [x] Add global modal structure (#global-modal with #strategy-modal, #user-modal, #delete-confirm)
- [x] Update #panel-call: Replace agent input with dropdown (#call-agent-select, required), add preview (#call-agent-preview), disable submit button
- [x] Update #panel-users: Add Edit/Delete buttons to table rows, move create form to modal, add "Create User" button
- [x] Add toast div (#toast)
- [x] Update inline <script>: Extend loadAgents() for all dropdowns, add event listeners for dropdown changes (fetch metadata, update previews, fetch strategies), modal open/close, form validations, add agent_id to payloads, show toasts

## 3. Update static/js/app.js (minor)
- [ ] Add showToast(msg) and animateElement(el, class) utilities if not inline

## 4. Update templates/admin_login.html (optional)
- [ ] Add "Create Admin" button opening modal with register form

## 5. Testing and Followup
- [x] Run server (python main.py), launch browser to /dashboard
- [x] Test dropdowns populate, agent select updates previews with animation
- [ ] Test modals open/close with animations, forms submit with agent_id
- [x] Test button disables, toasts appear
- [x] Debug any errors via console logs
- [x] Verify admin users modals work
