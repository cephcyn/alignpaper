import React from 'react';

import logo from './logo.svg';
import './App.css';

class App extends React.Component {
  componentDidMount() {
    const apiUrl = '/api?id=3';
    fetch(apiUrl)
      .then((response) => response.json())
      .then((data) => console.log('Hello data!:', data));
  }
  
  render() {
    return (
      <div className="App">
        <header className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
        </header>
      </div>
    );
  }
}

export default App;
