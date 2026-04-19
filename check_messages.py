import fallguard_app
from fallguard_app import app
with app.test_client() as c:
    with c.session_transaction() as sess:
        sess['username'] = 'nurse.anna'
        sess['role'] = 'Nurse'
        sess['name'] = 'Anna Wilson'
    rv = c.get('/nurse')
    html = rv.data.decode('utf-8')
    print('status', rv.status_code)
    print('contains /messages/', '/messages/' in html)
    print('contains Messages button', 'Messages' in html)
    idx = html.find('/messages/')
    print('first occurrence:', idx)
    if idx != -1:
        print(html[max(0, idx-120):idx+180])
