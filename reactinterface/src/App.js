import React from 'react';

import logo from './logo.svg';
import './App.css';

class AlignmentTable extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    console.log("rerendering AlignmentTable =========");
    console.log("props:", this.props);
    console.log("state:", this.state);

    const rows = this.props.data.map(
      (row) => {
        const cols = row.txt.map((cell, index) => {
          return <td key={index}>{cell.join(' ')}</td>
        });
        return (
          <tr key={row.id}>
            <td className="align-id">{row.id}</td>
            {cols}
          </tr>
        );
      }
    );

    return (
      <table>
        <tbody>
          {rows}
        </tbody>
      </table>
    );
  }
}

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {alignment: []};
  }

  componentDidMount() {
    fetch(this.props.apiUrl)
      .then((response) => response.json())
      .then((data) => this.setState(data));
  }

  render() {
    console.log("rerendering App =========");
    console.log("props:", this.props);
    console.log("state:", this.state);
    console.log(this.state.alignment);

    return (
      <div className="App">
        <AlignmentTable data={this.state.alignment} />
        <img src={logo} className="App-logo" alt="logo" />
      </div>
    );
  }
}

export default App;
