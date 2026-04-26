/**
 * Landing page auth integration.
 * Reads the 'genai-ess-auth' cookie (set by Lambda@Edge callback)
 * to determine if the user is authenticated. This cookie is NOT HttpOnly
 * so JS can read it for UI purposes. The actual auth token is in a
 * separate HttpOnly cookie that only Lambda@Edge can read.
 *
 * On localhost, buttons are always unlocked (no auth required for local dev).
 */
(function () {
  var isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  var authInfo = getAuthCookie();
  var isAuthenticated = isLocal || (authInfo && authInfo.exp && (authInfo.exp * 1000) > Date.now());

  var banner = document.getElementById('auth-banner');
  var btnSignin = document.getElementById('btn-signin');
  var btnSignout = document.getElementById('btn-signout');
  var headerUser = document.getElementById('header-user');
  var playgroundLinks = document.querySelectorAll('.playground-button');

  if (isAuthenticated && !isLocal) {
    // Authenticated state — show user info and unlock
    if (banner) banner.classList.add('hidden');
    if (btnSignin) btnSignin.style.display = 'none';
    if (btnSignout) btnSignout.style.display = 'inline-block';

    if (authInfo && authInfo.email) {
      var name = authInfo.email.split('@')[0];
      name = name.charAt(0).toUpperCase() + name.slice(1);
      if (headerUser) {
        headerUser.textContent = 'Welcome, ' + name;
        headerUser.style.display = 'inline';
      }
    }

    // Unlock all playground buttons
    playgroundLinks.forEach(function (link) {
      link.classList.remove('locked');
      link.innerHTML = 'Explore Playground <i class="fas fa-arrow-right" aria-hidden="true"></i>';
    });
  } else if (isLocal) {
    // Local dev — unlock everything, hide auth UI
    if (banner) banner.classList.add('hidden');
    if (btnSignin) btnSignin.style.display = 'none';
    if (btnSignout) btnSignout.style.display = 'none';
  }
  // else: default locked state from HTML (unauthenticated remote user)
})();

function getAuthCookie() {
  try {
    var match = document.cookie.match(/(?:^|;\s*)genai-ess-auth=([^;]*)/);
    if (!match) return null;
    var value = match[1];
    // Add base64 padding
    while (value.length % 4) value += '=';
    var decoded = atob(value.replace(/-/g, '+').replace(/_/g, '/'));
    return JSON.parse(decoded);
  } catch (e) {
    return null;
  }
}

function signIn() {
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return;
  }
  window.location.href = '/auth/login';
}
