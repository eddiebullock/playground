import hashlib

class User:
    def __init__(self, username, password):
        """Create new user object, pwd will be encrypted before storing."""
        self.username = username
        self.password = self._encrypted_pw(password)
        self.is_logged_in = False

    def _encrypted_pw(self, password):
        """Encrypt the password with username and return sha digest."""
        hash_string = (self.username + password)
        hash_string = hash_string.encode("utf8")
        return hashlib.sha256(hash_string).hexdigest()

    def check_password(self, password):
        """Check pw is valid for this user."""
        encrypted = self._encrypted_pw(password)
        return encrypted == self.password

class AuthException(Exception):
    def __init__(self, message):
        super().__init__(message)

class UsernameAlreadyExists(AuthException):
    pass

class PasswordTooShort(AuthException):
    pass

class InvalidUsername(AuthException):
    pass

class InvalidPassword(AuthException):
    pass

class PermissionError(AuthException):
    pass

class NotLoggedInError(AuthException):
    pass

class NotPermittedError(AuthException):
    pass

class Authenticator:
    def __init__(self):
        """Construct an auth to manage users coming in and out."""
        self.users = {}

    def add_user(self, username, password):
        if username in self.users:
            raise UsernameAlreadyExists(f"Username {username} already exists.")
        if len(password) < 6:
            raise PasswordTooShort("Password is too short.")
        self.users[username] = User(username, password)

    def login(self, username, password):
        try:
            user = self.users[username]
        except KeyError:
            raise InvalidUsername(username)
            
        if not user.check_password(password):
            raise InvalidPassword(username)
            
        user.is_logged_in = True
        return True

    def is_logged_in(self, username):
        if username in self.users:
            return self.users[username].is_logged_in
        return False

authenticator = Authenticator()

class Authorizor:
    def __init__(self, authenticator):
        self.authenticator = authenticator
        self.permissions = {}

    def add_permission(self, perm_name):
        """Create new permission that users can be added to."""
        if perm_name in self.permissions:
            raise PermissionError("Permission already exists.")
        self.permissions[perm_name] = set()

    def permit_user(self, perm_name, username):
        """Grant given permission to user."""
        try:
            perm_set = self.permissions[perm_name]
        except KeyError:
            raise PermissionError("Permission does not exist.")
            
        if username not in self.authenticator.users:
            raise InvalidUsername(username)
        perm_set.add(username)

    def check_permission(self, perm_name, username):
        if not self.authenticator.is_logged_in(username):
            raise NotLoggedInError(username)
        try:
            perm_set = self.permissions[perm_name]
        except KeyError:
            raise PermissionError("Permissions do not exist.")
        
        if username not in perm_set:
            raise NotPermittedError(username)
        return True

authorizor = Authorizor(authenticator)
