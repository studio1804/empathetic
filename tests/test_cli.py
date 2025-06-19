import pytest
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
from empathetic.cli import main, setup, env_check
import os
from pathlib import Path

class TestCLICommands:
    """Test CLI command functionality"""
    
    def test_cli_help(self):
        """Test that CLI help displays correctly"""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert 'Empathetic - AI Testing for Human Values' in result.output
        assert 'test' in result.output
        assert 'setup' in result.output
        
    def test_cli_version(self):
        """Test version command"""
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
        assert '0.1.0' in result.output
        
    @patch('empathetic.cli.asyncio.run')
    @patch('empathetic.core.tester.Tester')
    def test_test_command_basic(self, mock_tester_class, mock_asyncio_run):
        """Test basic test command functionality"""
        # Mock the tester and results
        mock_tester = Mock()
        mock_tester_class.return_value = mock_tester
        
        mock_results = Mock()
        mock_results.overall_score = 0.85
        mock_results.suite_results = {
            'bias': Mock(score=0.8, tests_passed=4, tests_total=5)
        }
        mock_results.recommendations = ['Test recommendation']
        
        mock_asyncio_run.return_value = mock_results
        
        runner = CliRunner()
        result = runner.invoke(main, ['test', 'gpt-3.5-turbo'])
        
        # Should succeed and show results
        assert result.exit_code == 0
        mock_tester.run_tests.assert_called_once()
        
    @patch('empathetic.cli.asyncio.run')
    @patch('empathetic.core.tester.Tester')
    def test_test_command_with_options(self, mock_tester_class, mock_asyncio_run):
        """Test test command with various options"""
        mock_tester = Mock()
        mock_tester_class.return_value = mock_tester
        
        mock_results = Mock()
        mock_results.overall_score = 0.95
        mock_results.suite_results = {}
        
        mock_asyncio_run.return_value = mock_results
        
        runner = CliRunner()
        result = runner.invoke(main, [
            'test', 'gpt-4', 
            '--suite', 'bias',
            '--suite', 'safety',
            '--threshold', '0.9',
            '--verbose'
        ])
        
        assert result.exit_code == 0
        call_args = mock_tester.run_tests.call_args
        assert 'bias' in call_args[1]['suites']
        assert 'safety' in call_args[1]['suites']
        assert call_args[1]['verbose'] == True
        
    @patch('empathetic.cli.asyncio.run')
    @patch('empathetic.core.tester.Tester')
    def test_test_command_failure(self, mock_tester_class, mock_asyncio_run):
        """Test test command when score below threshold"""
        mock_tester = Mock()
        mock_tester_class.return_value = mock_tester
        
        mock_results = Mock()
        mock_results.overall_score = 0.6  # Below default threshold of 0.9
        mock_results.suite_results = {}
        
        mock_asyncio_run.return_value = mock_results
        
        runner = CliRunner()
        result = runner.invoke(main, ['test', 'bad-model'])
        
        assert result.exit_code == 1  # Should exit with error code
        
    @patch('empathetic.cli.asyncio.run')
    @patch('empathetic.core.tester.Tester')
    def test_check_command(self, mock_tester_class, mock_asyncio_run):
        """Test check command functionality"""
        mock_tester = Mock()
        mock_tester_class.return_value = mock_tester
        
        mock_results = Mock()
        mock_results.suite_results = {
            'bias': Mock(score=0.85, tests_passed=4, tests_total=5)
        }
        
        mock_asyncio_run.return_value = mock_results
        
        runner = CliRunner()
        result = runner.invoke(main, ['check', 'gpt-3.5-turbo', '--suite', 'bias'])
        
        assert result.exit_code == 0
        assert 'bias' in result.output
        assert '0.85' in result.output or '0.850' in result.output

class TestSetupCommand:
    """Test interactive setup command"""
    
    @patch('empathetic.cli.Confirm.ask')
    @patch('empathetic.cli.Prompt.ask')
    @patch('builtins.open', create=True)
    def test_setup_basic_flow(self, mock_open, mock_prompt, mock_confirm):
        """Test basic setup flow with OpenAI key"""
        # Mock user inputs
        mock_confirm.side_effect = [True, False, False, False, False]  # Only overwrite, skip others
        mock_prompt.ask.return_value = 'sk-test1234567890'
        
        # Mock file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        runner = CliRunner()
        result = runner.invoke(setup)
        
        assert result.exit_code == 0
        assert 'Configuration saved' in result.output
        mock_file.write.assert_called()
        
    @patch('empathetic.cli.Confirm.ask')
    @patch('empathetic.cli.Prompt.ask')
    def test_setup_cancelled(self, mock_prompt, mock_confirm):
        """Test setup cancellation when .env exists"""
        mock_confirm.return_value = False  # Don't overwrite
        
        with patch('pathlib.Path.exists', return_value=True):
            runner = CliRunner()
            result = runner.invoke(setup)
            
            assert result.exit_code == 0
            assert 'cancelled' in result.output
            
    @patch('empathetic.cli.Confirm.ask')
    @patch('empathetic.cli.Prompt.ask')
    def test_setup_force_flag(self, mock_prompt, mock_confirm):
        """Test setup with force flag"""
        mock_prompt.ask.return_value = 'sk-test1234567890'
        mock_confirm.side_effect = [False, False, False, False]  # Skip optional configs
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', create=True) as mock_open:
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                runner = CliRunner()
                result = runner.invoke(setup, ['--force'])
                
                assert result.exit_code == 0
                # Should not ask about overwriting with force flag
                assert 'Configuration saved' in result.output
                
    @patch('empathetic.cli.Confirm.ask')
    @patch('empathetic.cli.Prompt.ask')
    @patch('builtins.open', create=True)
    def test_setup_all_providers(self, mock_open, mock_prompt, mock_confirm):
        """Test setup with all providers configured"""
        # Mock user inputs for all providers
        mock_confirm.side_effect = [True, True, True, False, False]  # Overwrite, Anthropic, HF, skip others
        mock_prompt.side_effect = [
            'sk-openai123',      # OpenAI key
            'sk-ant-anthropic456', # Anthropic key  
            'hf_huggingface789'   # HuggingFace key
        ]
        
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        runner = CliRunner()
        result = runner.invoke(setup)
        
        assert result.exit_code == 0
        # Check that all keys were written
        write_calls = [call[0][0] for call in mock_file.write.call_args_list]
        full_content = ''.join(write_calls)
        assert 'OPENAI_API_KEY=sk-openai123' in full_content
        assert 'ANTHROPIC_API_KEY=sk-ant-anthropic456' in full_content
        assert 'HUGGINGFACE_API_KEY=hf_huggingface789' in full_content

class TestEnvCheckCommand:
    """Test environment check command"""
    
    def test_env_check_no_keys(self):
        """Test env check with no API keys set"""
        with patch.dict(os.environ, {}, clear=True):
            runner = CliRunner()
            result = runner.invoke(env_check)
            
            assert result.exit_code == 0
            assert 'Environment Check' in result.output
            assert 'Missing' in result.output or 'Not set' in result.output
            
    def test_env_check_with_keys(self):
        """Test env check with API keys set"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'sk-test123',
            'ANTHROPIC_API_KEY': 'sk-ant-test456'
        }):
            runner = CliRunner()
            result = runner.invoke(env_check)
            
            assert result.exit_code == 0
            assert 'sk-test12...' in result.output  # Masked key
            assert 'Set' in result.output
            
    @patch('pathlib.Path.exists')
    def test_env_check_with_files(self, mock_exists):
        """Test env check detects .env and config files"""
        # Mock file existence
        def mock_exists_side_effect(path_obj):
            path_str = str(path_obj)
            if '.env' in path_str:
                return True
            elif 'config/default.yaml' in path_str:
                return True
            return False
            
        mock_exists.side_effect = mock_exists_side_effect
        
        runner = CliRunner()
        result = runner.invoke(env_check)
        
        assert result.exit_code == 0
        assert '.env file found' in result.output
        assert 'Config file found' in result.output

class TestReportCommand:
    """Test report generation command"""
    
    @patch('empathetic.reports.generator.ReportGenerator')
    def test_report_command_no_input(self, mock_generator_class):
        """Test report command without input file"""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        runner = CliRunner()
        result = runner.invoke(main, ['report'])
        
        assert result.exit_code == 0
        assert 'No input file specified' in result.output
        
    @patch('empathetic.reports.generator.ReportGenerator')
    def test_report_command_with_input(self, mock_generator_class):
        """Test report command with input file"""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_generator.create_from_file.return_value = "Report generated"
        
        runner = CliRunner()
        result = runner.invoke(main, ['report', '--input', 'test.json', '--format', 'html'])
        
        assert result.exit_code == 0
        mock_generator.create_from_file.assert_called_once_with('test.json', format='html')

class TestCLIEdgeCases:
    """Test CLI edge cases and error handling"""
    
    def test_invalid_suite_name(self):
        """Test handling of invalid test suite names"""
        runner = CliRunner()
        result = runner.invoke(main, ['check', 'gpt-3.5-turbo', '--suite', 'invalid_suite'])
        
        # Should handle gracefully (specific behavior depends on implementation)
        assert result.exit_code in [0, 1]
        
    @patch('empathetic.cli.asyncio.run')
    def test_test_command_exception(self, mock_asyncio_run):
        """Test test command when an exception occurs"""
        mock_asyncio_run.side_effect = Exception("Test error")
        
        runner = CliRunner()
        result = runner.invoke(main, ['test', 'gpt-3.5-turbo'])
        
        assert result.exit_code == 1
        assert 'Error' in result.output
        
    def test_validate_command_placeholder(self):
        """Test validate command (currently placeholder)"""
        runner = CliRunner()
        result = runner.invoke(main, ['validate', '/some/path'])
        
        assert result.exit_code == 0
        assert 'not yet implemented' in result.output
        
    @patch('builtins.open', side_effect=IOError("Permission denied"))
    @patch('empathetic.cli.Confirm.ask', return_value=True)
    @patch('empathetic.cli.Prompt.ask', return_value='sk-test123')
    def test_setup_file_error(self, mock_prompt, mock_confirm, mock_open):
        """Test setup command when file writing fails"""
        runner = CliRunner()
        result = runner.invoke(setup, ['--force'])
        
        assert result.exit_code == 1
        assert 'Error writing .env file' in result.output

class TestCLIIntegration:
    """Integration tests for CLI commands"""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
    @patch('empathetic.core.tester.Tester')
    @patch('empathetic.cli.asyncio.run')
    def test_full_test_workflow(self, mock_asyncio_run, mock_tester_class):
        """Test complete test workflow"""
        # Setup mock results
        mock_tester = Mock()
        mock_tester_class.return_value = mock_tester
        
        mock_results = Mock()
        mock_results.overall_score = 0.87
        mock_results.suite_results = {
            'bias': Mock(score=0.85, tests_passed=4, tests_total=5, recommendations=[]),
            'safety': Mock(score=0.89, tests_passed=7, tests_total=8, recommendations=[])
        }
        mock_results.recommendations = ['Overall recommendation']
        
        mock_asyncio_run.return_value = mock_results
        
        runner = CliRunner()
        result = runner.invoke(main, [
            'test', 'gpt-3.5-turbo',
            '--suite', 'bias',
            '--suite', 'safety', 
            '--verbose',
            '--threshold', '0.8'
        ])
        
        assert result.exit_code == 0
        assert '0.87' in result.output or '0.870' in result.output
        assert 'bias' in result.output
        assert 'safety' in result.output
        assert 'Passed' in result.output  # Score above threshold