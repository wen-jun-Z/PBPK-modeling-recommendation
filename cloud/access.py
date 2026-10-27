"""Optional local-preview gate; Community Cloud access is managed by its platform."""
import hashlib
import hmac
import os


def enforce_preview_access():
    expected = os.environ.get('PBPK_PREVIEW_ACCESS_HASH', '')
    if not expected:
        return
    import streamlit as st
    if st.session_state.get('_preview_authorized_hash') == expected:
        return
    st.title('PBPK private preview')
    with st.form('private_preview', clear_on_submit=True):
        code = st.text_input('Preview access code', type='password')
        submitted = st.form_submit_button('Open preview')
    if submitted:
        if hmac.compare_digest(hashlib.sha256(code.encode()).hexdigest(), expected):
            st.session_state['_preview_authorized_hash'] = expected
            st.rerun()
        st.error('Incorrect access code')
    st.stop()
