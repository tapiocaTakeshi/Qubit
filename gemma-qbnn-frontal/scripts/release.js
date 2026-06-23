#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function question(query) {
  return new Promise((resolve) => rl.question(query, resolve));
}

async function main() {
  console.log('\n🚀 Gemma QBNN Frontal - Manual Release Script\n');

  const packageJsonPath = path.join(__dirname, '../package.json');
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf-8'));
  const currentVersion = packageJson.version;

  console.log(`Current version: ${currentVersion}`);
  console.log('Release type:');
  console.log('  1. Patch (1.0.0 → 1.0.1)');
  console.log('  2. Minor (1.0.0 → 1.1.0)');
  console.log('  3. Major (1.0.0 → 2.0.0)');
  console.log('  4. Custom version');

  const choice = await question('\nSelect option (1-4): ');

  let newVersion;
  if (choice === '1') {
    const parts = currentVersion.split('.');
    parts[2] = String(parseInt(parts[2]) + 1);
    newVersion = parts.join('.');
  } else if (choice === '2') {
    const parts = currentVersion.split('.');
    parts[1] = String(parseInt(parts[1]) + 1);
    parts[2] = '0';
    newVersion = parts.join('.');
  } else if (choice === '3') {
    const parts = currentVersion.split('.');
    parts[0] = String(parseInt(parts[0]) + 1);
    parts[1] = '0';
    parts[2] = '0';
    newVersion = parts.join('.');
  } else if (choice === '4') {
    newVersion = await question('Enter custom version (e.g., 1.0.1): ');
  } else {
    console.error('Invalid choice. Exiting.');
    process.exit(1);
  }

  console.log(`\n📝 New version: ${newVersion}`);
  const confirm = await question('Continue with release? (y/n): ');

  if (confirm.toLowerCase() !== 'y') {
    console.log('Release cancelled.');
    process.exit(0);
  }

  try {
    console.log('\n📦 Building package...');
    execSync('npm run build', { cwd: path.join(__dirname, '..'), stdio: 'inherit' });

    console.log('\n✅ Running tests...');
    try {
      execSync('npm test', { cwd: path.join(__dirname, '..'), stdio: 'inherit' });
    } catch (e) {
      console.warn('⚠️  Tests failed but continuing...');
    }

    console.log('\n🔧 Updating version...');
    execSync(`npm version ${newVersion} --no-git-tag-version`, {
      cwd: path.join(__dirname, '..'),
      stdio: 'inherit',
    });

    console.log('\n📤 Publishing to npm...');
    console.log('Note: You must be logged in to npm. Run "npm login" if needed.\n');
    execSync('npm publish --access public', {
      cwd: path.join(__dirname, '..'),
      stdio: 'inherit',
    });

    console.log('\n✨ Package published successfully!\n');

    const gitTag = await question('Create git tag and commit? (y/n): ');
    if (gitTag.toLowerCase() === 'y') {
      const cwd = path.join(__dirname, '../..');

      execSync('git add gemma-qbnn-frontal/package.json gemma-qbnn-frontal/package-lock.json', {
        cwd,
        stdio: 'inherit',
      });

      execSync(`git commit -m "chore: bump version to ${newVersion}"`, {
        cwd,
        stdio: 'inherit',
      });

      execSync(`git tag gemma-qbnn-frontal@${newVersion}`, {
        cwd,
        stdio: 'inherit',
      });

      console.log(`\n📌 Git tag created: gemma-qbnn-frontal@${newVersion}`);
      console.log('Push with: git push origin && git push origin gemma-qbnn-frontal@' + newVersion);

      const pushNow = await question('Push to remote now? (y/n): ');
      if (pushNow.toLowerCase() === 'y') {
        execSync('git push origin', { cwd, stdio: 'inherit' });
        execSync(`git push origin gemma-qbnn-frontal@${newVersion}`, {
          cwd,
          stdio: 'inherit',
        });
        console.log('\n✅ Pushed to remote!');
      }
    }

    console.log(`\n🎉 Release complete!`);
    console.log(`📦 Package: https://www.npmjs.com/package/gemma-qbnn-frontal@${newVersion}`);

    rl.close();
  } catch (error) {
    console.error('\n❌ Error during release:', error.message);
    rl.close();
    process.exit(1);
  }
}

main();
