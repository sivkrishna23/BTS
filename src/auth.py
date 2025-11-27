"""
Authentication and User Management Module
Handles registration, login, student verification, and subscription management
"""

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime, timedelta
import re

class AuthManager:
    def __init__(self, data_dir="user_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.users_file = self.data_dir / "users.json"
        self.sessions_file = self.data_dir / "sessions.json"
        
        # Initialize files if they don't exist
        if not self.users_file.exists():
            self._save_json(self.users_file, {})
        if not self.sessions_file.exists():
            self._save_json(self.sessions_file, {})
    
    def _save_json(self, filepath, data):
        """Save data to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_json(self, filepath):
        """Load data from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _hash_password(self, password):
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _validate_email(self, email):
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _is_student_email(self, email):
        """Check if email is from educational institution."""
        edu_domains = ['.edu', '.ac.', 'student', 'university', 'college']
        return any(domain in email.lower() for domain in edu_domains)
    
    def register_user(self, username, email, password, user_type='regular', 
                     college_name=None, college_id_path=None):
        """
        Register a new user.
        
        Args:
            username: User's username
            email: User's email
            password: User's password
            user_type: 'student' or 'regular'
            college_name: Name of college (for students)
            college_id_path: Path to college ID image (for students)
        
        Returns:
            dict with 'success' and 'message'
        """
        users = self._load_json(self.users_file)
        
        # Validation
        if username in users:
            return {'success': False, 'message': 'Username already exists'}
        
        if not self._validate_email(email):
            return {'success': False, 'message': 'Invalid email format'}
        
        if any(u['email'] == email for u in users.values()):
            return {'success': False, 'message': 'Email already registered'}
        
        if len(password) < 6:
            return {'success': False, 'message': 'Password must be at least 6 characters'}
        
        # Student verification
        if user_type == 'student':
            if not college_name or not college_id_path:
                return {'success': False, 'message': 'College name and ID required for student registration'}
            
            if not self._is_student_email(email):
                return {'success': False, 'message': 'Please use your college email address'}
        
        # Create user
        user_data = {
            'username': username,
            'email': email,
            'password': self._hash_password(password),
            'user_type': user_type,
            'subscription_status': 'free' if user_type == 'student' else 'trial',
            'registration_date': datetime.now().isoformat(),
            'trial_end_date': (datetime.now() + timedelta(days=7)).isoformat() if user_type == 'regular' else None,
            'college_name': college_name if user_type == 'student' else None,
            'college_id_verified': False if user_type == 'student' else None,
            'college_id_path': str(college_id_path) if college_id_path else None
        }
        
        users[username] = user_data
        self._save_json(self.users_file, users)
        
        return {'success': True, 'message': f'Registration successful! {"Student account created (free)" if user_type == "student" else "7-day trial started"}'}
    
    def login(self, username, password):
        """
        Authenticate user.
        
        Returns:
            dict with 'success', 'message', and 'user_data'
        """
        users = self._load_json(self.users_file)
        
        if username not in users:
            return {'success': False, 'message': 'Invalid username or password'}
        
        user = users[username]
        
        if user['password'] != self._hash_password(password):
            return {'success': False, 'message': 'Invalid username or password'}
        
        # Check subscription status
        if user['user_type'] == 'regular' and user['subscription_status'] == 'trial':
            trial_end = datetime.fromisoformat(user['trial_end_date'])
            if datetime.now() > trial_end:
                user['subscription_status'] = 'expired'
                users[username] = user
                self._save_json(self.users_file, users)
                return {'success': False, 'message': 'Trial expired. Please subscribe ($10.99/month)'}
        
        # Create session
        session_id = hashlib.sha256(f"{username}{datetime.now()}".encode()).hexdigest()
        sessions = self._load_json(self.sessions_file)
        sessions[session_id] = {
            'username': username,
            'login_time': datetime.now().isoformat()
        }
        self._save_json(self.sessions_file, sessions)
        
        return {
            'success': True,
            'message': 'Login successful',
            'user_data': user,
            'session_id': session_id
        }
    
    def get_user_info(self, username):
        """Get user information."""
        users = self._load_json(self.users_file)
        return users.get(username)
    
    def verify_student_id(self, username):
        """Mark student ID as verified (admin function)."""
        users = self._load_json(self.users_file)
        if username in users and users[username]['user_type'] == 'student':
            users[username]['college_id_verified'] = True
            self._save_json(self.users_file, users)
            return True
        return False
    
    def subscribe(self, username):
        """Subscribe user to paid plan."""
        users = self._load_json(self.users_file)
        if username in users:
            users[username]['subscription_status'] = 'active'
            users[username]['subscription_date'] = datetime.now().isoformat()
            self._save_json(self.users_file, users)
            return True
        return False
    
    # ============================================================================
    # ADMIN FUNCTIONS
    # ============================================================================
    
    ADMIN_USERNAME = "Aakisetty"
    ADMIN_PASSWORD_HASH = hashlib.sha256("Sivakrishna23@".encode()).hexdigest()
    
    def admin_login(self, username, password):
        """Admin login with hardcoded credentials."""
        if username == self.ADMIN_USERNAME and self._hash_password(password) == self.ADMIN_PASSWORD_HASH:
            return {
                'success': True,
                'message': 'Admin login successful',
                'is_admin': True,
                'user_data': {
                    'username': username,
                    'user_type': 'admin',
                    'subscription_status': 'unlimited'
                }
            }
        return {'success': False, 'message': 'Invalid admin credentials'}
    
    def get_all_users(self):
        """Get all users (admin only)."""
        users = self._load_json(self.users_file)
        return users
    
    def block_user(self, username):
        """Block a user (admin only)."""
        users = self._load_json(self.users_file)
        if username in users:
            users[username]['blocked'] = True
            users[username]['blocked_date'] = datetime.now().isoformat()
            self._save_json(self.users_file, users)
            return True
        return False
    
    def unblock_user(self, username):
        """Unblock a user (admin only)."""
        users = self._load_json(self.users_file)
        if username in users:
            users[username]['blocked'] = False
            self._save_json(self.users_file, users)
            return True
        return False
    
    def delete_user(self, username):
        """Delete a user (admin only)."""
        users = self._load_json(self.users_file)
        if username in users:
            del users[username]
            self._save_json(self.users_file, users)
            return True
        return False
    
    def update_user_subscription(self, username, status):
        """Update user subscription status (admin only)."""
        users = self._load_json(self.users_file)
        if username in users:
            users[username]['subscription_status'] = status
            self._save_json(self.users_file, users)
            return True
        return False
    
    def get_user_stats(self):
        """Get user statistics (admin only)."""
        users = self._load_json(self.users_file)
        stats = {
            'total_users': len(users),
            'students': sum(1 for u in users.values() if u['user_type'] == 'student'),
            'professionals': sum(1 for u in users.values() if u['user_type'] == 'regular'),
            'active_subscriptions': sum(1 for u in users.values() if u.get('subscription_status') == 'active'),
            'blocked_users': sum(1 for u in users.values() if u.get('blocked', False)),
            'pending_verifications': sum(1 for u in users.values() if u['user_type'] == 'student' and not u.get('college_id_verified', False))
        }
        return stats

